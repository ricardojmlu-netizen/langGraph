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
KNOWLEDGE = {
    "architecture": "StateGraph 用节点与边编排；节点返回部分状态；reducer 合并并发写入。",
    "reliability": "Checkpointer 按 thread_id 保存状态；interrupt 与 Command 支持暂停恢复。",
    "operations": "stream 输出更新；get_state_history 检查历史；update_state 创建分支。",
}


@tool
def search_notes(topic: str) -> str:
    """Read local tutorial notes for architecture, reliability, or operations."""
    return KNOWLEDGE.get(topic, "未知主题。可选 architecture、reliability、operations。")


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    topic: str
    question: str
    model_turns: int


class State(TypedDict):
    question: str
    topics: list[str]
    findings: Annotated[list[dict], operator.add]
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
    """显式 ReAct 循环：model → ToolNode → model；子图拥有独立消息状态。"""
    model = None
    if real_model:
        from .model import deepseek_model

        model = deepseek_model().bind_tools([search_notes])

    def reason(state: ResearchState):
        turns = state.get("model_turns", 0)
        if turns >= 4:
            return {
                "messages": [AIMessage(content="工具调用达到上限，请根据已有资料继续。")],
                "model_turns": turns + 1,
            }
        if model is not None:
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
            reply = AIMessage(content=f"[{state['topic']}] {state['messages'][-1].content}")
        return {"messages": [reply], "model_turns": turns + 1}

    builder = StateGraph(ResearchState)
    builder.add_node(
        "model",
        reason,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, retry_on=ConnectionError),
    )
    builder.add_node("tools", ToolNode([search_notes], handle_tool_errors=True))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return builder.compile()


def build_graph(checkpointer=None, store=None, real_model: bool = False):
    researcher = research_graph(real_model)

    def plan(state: State, runtime: Runtime[Context]):
        if not state["question"].strip():
            raise ValueError("question 不能为空")
        saved = (
            runtime.store.get(("preferences", runtime.context.user_id), "style")
            if runtime.store
            else None
        )
        preference = saved.value["text"] if saved else runtime.context.preference
        get_stream_writer()({"event": "planned", "topics": state["topics"]})
        return {"preference": preference, "messages": [HumanMessage(content=state["question"])]}

    def dispatch(state: State):
        return [
            Send("research", {"topic": topic, "question": state["question"]})
            for topic in state["topics"]
        ]

    def research(state: ResearchState):
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
        ordered = sorted(state["findings"], key=lambda item: item["topic"])
        report = f"# {state['question']}\n\n输出偏好：{state['preference']}\n\n"
        report += "\n\n".join(f"## {item['topic']}\n{item['text']}" for item in ordered)
        report += "\n\n实施建议：先运行离线流程，再接入模型；用独立 thread_id 隔离会话。"
        return {"report": report}

    def review(state: State, runtime: Runtime[Context]) -> Command[Literal["remember", "rejected"]]:
        decision = (
            interrupt({"kind": "review", "report": state["report"], "expected": {"approved": True}})
            if runtime.context.human_review
            else {"approved": True}
        )
        if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
            raise ValueError("恢复值必须为包含布尔 approved 的 JSON 对象")
        approved = decision["approved"]
        return Command(update={"approved": approved}, goto="remember" if approved else "rejected")

    def remember(state: State, runtime: Runtime[Context]):
        if runtime.store:
            # 确定性 key 的 put 可重复执行，避免恢复时重复追加副作用。
            runtime.store.put(
                ("preferences", runtime.context.user_id), "style", {"text": state["preference"]}
            )
        return {"messages": [AIMessage(content=state["report"])]}

    def rejected(state: State):
        return {"messages": [AIMessage(content="报告未通过审核，未保存偏好。")]}

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
    graph.add_conditional_edges("plan", dispatch, ["research"])
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", "review")
    graph.add_edge("remember", END)
    graph.add_edge("rejected", END)
    return graph.compile(checkpointer=checkpointer, store=store)


def initial_state(question: str, topics: list[str] | None = None):
    selected = list(dict.fromkeys(topics if topics is not None else KNOWLEDGE))
    if not selected or any(topic not in KNOWLEDGE for topic in selected):
        raise ValueError(f"topics 必须是 {list(KNOWLEDGE)} 的非空子集")
    return {"question": question, "topics": selected, "findings": [], "messages": []}
