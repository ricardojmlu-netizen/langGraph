"""研究助手：状态 → 规划 → Send 并行 → 子图 Agent → 汇总 → 审核 → 记忆。"""

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, Send, interrupt

# 小型本地知识库，刻意不把离线检索伪装成联网搜索。
# Small local knowledge base; deliberately never presented as internet search results.
KNOWLEDGE = {
    "architecture": "StateGraph 用节点与边编排；节点返回部分状态；reducer 合并并发写入。",
    "reliability": "Checkpointer 按 thread_id 保存状态；interrupt 与 Command 支持暂停恢复。",
    "operations": "stream 输出更新；get_state_history 检查历史；update_state 创建分支。",
}


@tool
def search_notes(topic: str) -> str:
    """读取本地教程笔记 / Read local tutorial notes for a supported topic."""
    return KNOWLEDGE.get(topic, "未知主题。可选 architecture、reliability、operations。")


class ResearchState(TypedDict):
    # add_messages 根据消息 ID 合并更新，适合工具调用形成的消息历史。
    # add_messages merges updates by message ID, which suits tool-call message history.
    messages: Annotated[list[AnyMessage], add_messages]
    topic: str
    question: str
    model_turns: int


class State(TypedDict):
    question: str
    topics: list[str]
    # 并行研究节点同时写入 findings，因此使用加法 reducer 累积，而不是相互覆盖。
    # Parallel research nodes write findings concurrently, so an additive reducer accumulates them.
    findings: Annotated[list[dict], operator.add]
    # 主图也保留消息轨迹，供最终回复和调试使用。
    # The parent graph also retains message history for the final reply and debugging.
    messages: Annotated[list[AnyMessage], add_messages]
    report: str
    approved: bool
    preference: str


@dataclass
class Context:
    user_id: str = "demo-user"
    human_review: bool = False
    preference: str = "中文，简洁，包含实施建议"


def research_graph(real_model: bool = False):
    """显式 ReAct 循环 / Explicit ReAct loop: model → ToolNode → model."""
    model = None
    if real_model:
        from .model import deepseek_model

        # bind_tools 将工具 schema 交给 DeepSeek；模型只决定是否调用，ToolNode 负责执行。
        # bind_tools exposes the tool schema to DeepSeek; the model decides, ToolNode executes.
        model = deepseek_model().bind_tools([search_notes])

    def reason(state: ResearchState):
        turns = state.get("model_turns", 0)
        if turns >= 4:
            # 双重上限：此处限制模型轮次，CLI 的 recursion_limit 限制整个图的超级步数。
            # Two limits: this caps model turns, while CLI recursion_limit caps graph super-steps.
            return {
                "messages": [AIMessage(content="工具调用达到上限，请根据已有资料继续。")],
                "model_turns": turns + 1,
            }
        if model is not None:
            # 真实模式保留完整消息历史，使工具结果回到同一个模型决策循环。
            # Real mode keeps full history so tool results return to the same model decision loop.
            reply = model.invoke(
                [
                    SystemMessage(
                        content="你是研究员。先调用本地 search_notes，再基于结果回答，"
                        "不要声称查过互联网。"
                    ),
                    *state["messages"],
                ]
            )
        elif turns == 0:
            # 离线模式构造一次真实的 tool call，以演示相同的 ToolNode 执行路径。
            # Offline mode creates a real tool call to demonstrate the same ToolNode path.
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(uuid4()),
                        "name": "search_notes",
                        "args": {"topic": state["topic"]},
                        "type": "tool_call",
                    }
                ],
            )
        else:
            # 工具已返回时，离线模式以工具结果生成确定性的研究结论。
            # After the tool returns, offline mode produces a deterministic finding from it.
            reply = AIMessage(content=f"[{state['topic']}] {state['messages'][-1].content}")
        return {"messages": [reply], "model_turns": turns + 1}

    # 子图与父图分离状态：子图只关心专题、消息和模型轮次。
    # The subgraph isolates state: it only needs topic, messages, and model turns.
    builder = StateGraph(ResearchState)
    builder.add_node(
        "model",
        reason,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, retry_on=ConnectionError),
    )
    # ToolNode 将 tool_calls 转为实际函数调用；错误会作为工具消息返回给模型。
    # ToolNode executes tool_calls; errors become tool messages returned to the model.
    builder.add_node("tools", ToolNode([search_notes], handle_tool_errors=True))
    builder.add_edge(START, "model")
    # 条件边：有 tool_calls 则转 tools，否则结束子图。
    # Conditional edge: route to tools when tool_calls exist; otherwise finish the subgraph.
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return builder.compile()


def build_graph(checkpointer=None, store=None, real_model: bool = False):
    # 每个父图研究节点复用同一套子图定义；LangGraph 为每次调用隔离执行状态。
    # Parent research nodes reuse one subgraph definition; LangGraph isolates each invocation state.
    researcher = research_graph(real_model)

    def plan(state: State, runtime: Runtime[Context]):
        if not state["question"].strip():
            raise ValueError("question 不能为空")
        # Store 是跨 thread 的长期记忆；checkpointer 则保存单次 thread 的执行状态。
        # Store is cross-thread long-term memory; the checkpointer saves one thread's execution state.
        saved = (
            runtime.store.get(("preferences", runtime.context.user_id), "style")
            if runtime.store
            else None
        )
        preference = saved.value["text"] if saved else runtime.context.preference
        # custom stream 发送业务进度，不暴露完整内部 state。
        # The custom stream emits business progress without exposing the full internal state.
        get_stream_writer()({"event": "planned", "topics": state["topics"]})
        return {"preference": preference, "messages": [HumanMessage(content=state["question"])]}

    def dispatch(state: State):
        # Send 为每个专题生成独立任务，LangGraph 在同一超级步并行调度它们。
        # Send creates an independent task per topic; LangGraph schedules them in parallel.
        return [
            Send("research", {"topic": topic, "question": state["question"]})
            for topic in state["topics"]
        ]

    def research(state: ResearchState):
        # 显式映射父图输入到子图，并只将最终结论映射回父图 findings。
        # Explicitly map parent input to the subgraph and map only its final finding back.
        result = researcher.invoke(
            {
                "topic": state["topic"],
                "question": state["question"],
                "model_turns": 0,
                "messages": [
                    HumanMessage(content=f"问题：{state['question']}；专题：{state['topic']}")
                ],
            }
        )
        get_stream_writer()({"event": "research_done", "topic": state["topic"]})
        return {"findings": [{"topic": state["topic"], "text": result["messages"][-1].content}]}

    def synthesize(state: State):
        # 按专题排序，避免并行完成顺序造成不稳定的报告内容。
        # Sort by topic so parallel completion order cannot make reports nondeterministic.
        ordered = sorted(state["findings"], key=lambda item: item["topic"])
        report = f"# {state['question']}\n\n输出偏好：{state['preference']}\n\n"
        report += "\n\n".join(f"## {item['topic']}\n{item['text']}" for item in ordered)
        report += "\n\n实施建议：先运行离线流程，再接入模型；用独立 thread_id 隔离会话。"
        return {"report": report}

    def review(state: State, runtime: Runtime[Context]) -> Command[Literal["remember", "rejected"]]:
        # interrupt 会将当前状态交给 checkpointer 并暂停；resume 值成为 decision。
        # interrupt checkpoints current state and pauses; the resume value becomes decision.
        decision = (
            interrupt({"kind": "review", "report": state["report"], "expected": {"approved": True}})
            if runtime.context.human_review
            else {"approved": True}
        )
        if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
            raise ValueError("恢复值必须为包含布尔 approved 的 JSON 对象")
        approved = decision["approved"]
        # Command 在一次节点返回中同时写状态并选择后继分支。
        # Command updates state and selects the next branch in a single node return.
        return Command(update={"approved": approved}, goto="remember" if approved else "rejected")

    def remember(state: State, runtime: Runtime[Context]):
        if runtime.store:
            # 确定性 key 的 put 可重复执行，避免恢复时重复追加副作用。
            # A deterministic key makes put replay-safe instead of appending duplicate side effects.
            runtime.store.put(
                ("preferences", runtime.context.user_id), "style", {"text": state["preference"]}
            )
        return {"messages": [AIMessage(content=state["report"])]}

    def rejected(state: State):
        return {"messages": [AIMessage(content="报告未通过审核，未保存偏好。")]}

    # Context 通过运行时注入，避免把用户身份、审核开关等运行参数写入可恢复 state。
    # Context is injected at runtime, keeping user identity and review flags out of resumable state.
    graph = StateGraph(State, context_schema=Context)
    for name, node in [
        ("plan", plan),
        ("research", research),
        ("synthesize", synthesize),
        ("review", review),
        ("remember", remember),
        ("rejected", rejected),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "plan")
    # map-reduce：plan 动态扇出研究任务，所有 research 完成后再汇总。
    # Map-reduce: plan fans out research tasks, then synthesizes after every research task completes.
    graph.add_conditional_edges("plan", dispatch, ["research"])
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", "review")
    graph.add_edge("remember", END)
    graph.add_edge("rejected", END)
    # 在编译时注入可选持久化组件；内存实现便于示例，SQLite 适合 CLI 恢复演示。
    # Inject optional persistence at compile time; memory is for demos, SQLite supports CLI recovery.
    return graph.compile(checkpointer=checkpointer, store=store)


def initial_state(question: str, topics: list[str] | None = None):
    # 去重并验证输入，使 Send 不会创建未知或重复的研究任务。
    # Deduplicate and validate input so Send never creates unknown or duplicate research tasks.
    selected = list(dict.fromkeys(topics if topics is not None else KNOWLEDGE))
    if not selected or any(topic not in KNOWLEDGE for topic in selected):
        raise ValueError(f"topics 必须是 {list(KNOWLEDGE)} 的非空子集")
    return {"question": question, "topics": selected, "findings": [], "messages": []}
