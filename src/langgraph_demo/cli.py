"""CLI checkpoints persist across processes; the example Store is process-local."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from .graph import Context, build_graph, initial_state


def dump(value):
    print(json.dumps(value, ensure_ascii=False, default=str, indent=2))


def main():
    parser = argparse.ArgumentParser(description="LangGraph 研究助手能力演示")
    parser.add_argument(
        "action",
        choices=["run", "resume", "state", "history", "fork", "diagram"],
        nargs="?",
        default="run",
    )
    parser.add_argument("--question", default="如何用 LangGraph 构建可靠的 Agent？")
    parser.add_argument("--thread", help="run 默认生成新会话，其余操作必须指定")
    parser.add_argument("--db", default=".demo/checkpoints.sqlite")
    parser.add_argument("--review", action="store_true", help="可选：演示人工审核中断")
    parser.add_argument("--reject", action="store_true", help="resume 时拒绝报告")
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--stream", choices=["updates", "values", "custom", "messages", "debug"])
    parser.add_argument("--checkpoint", help="fork 所基于的 checkpoint_id")
    parser.add_argument("--report", help="fork 时替换的报告文本")
    args = parser.parse_args()
    if args.action not in {"run", "diagram"} and not args.thread:
        parser.error("此操作需要 --thread")
    if args.action == "fork" and (not args.checkpoint or not args.report):
        parser.error("fork 需要 --checkpoint 和 --report")
    # 每次新 run 分配独立 thread_id；恢复、历史和分支必须显式指定已有 thread。
    # Each new run gets a distinct thread_id; resume, history, and fork require an existing thread.
    thread = args.thread or str(uuid4())
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    # SQLite checkpointer 在进程退出后仍保存状态，因此可用第二次 CLI 调用恢复。
    # SQLite checkpoints survive process exit, so a later CLI invocation can resume work.
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(args.db) as saver:
        graph = build_graph(saver, InMemoryStore(), args.real_model)
        if args.action == "diagram":
            print(graph.get_graph().draw_mermaid())
            return
        print(f"thread_id={thread}")
        snapshot = graph.get_state(config)
        if args.action == "state":
            dump({"values": snapshot.values, "next": snapshot.next, "tasks": snapshot.tasks})
            return
        if args.action == "history":
            # history 返回不可变 checkpoint 快照，供检查与后续时间旅行使用。
            # history returns immutable checkpoint snapshots for inspection and time travel.
            for item in graph.get_state_history(config):
                dump(
                    {
                        "checkpoint": item.config["configurable"]["checkpoint_id"],
                        "next": item.next,
                        "step": item.metadata.get("step"),
                    }
                )
            return
        if args.action == "run" and snapshot.values:
            parser.error("thread 已存在；请用新 thread，或 resume/fork，避免 reducer 累加旧结果")
        if args.action == "resume" and not any(t.interrupts for t in snapshot.tasks):
            parser.error("此 thread 没有等待恢复的 interrupt")
        if args.action == "fork":
            # 指定旧 checkpoint 后更新 report；这会创建新历史分支，而不修改旧快照。
            # Select an old checkpoint then update report; this creates a new branch without mutation.
            config["configurable"]["checkpoint_id"] = args.checkpoint
            if not graph.get_state(config).values:
                parser.error("checkpoint 不存在")
            config = graph.update_state(config, {"report": args.report}, as_node="synthesize")
            payload = None
            print("已从历史 checkpoint 创建新分支，原历史保留")
        elif args.action == "resume":
            # Command(resume=...) 将人工决策送回上次 interrupt 的调用位置。
            # Command(resume=...) sends the human decision back to the interrupted call site.
            payload = Command(resume={"approved": not args.reject})
        else:
            payload = initial_state(args.question)
        # Context 不在 checkpoint 中持久化，恢复时显式重新提供。
        # Context is not checkpointed, so provide it again when resuming.
        context = Context(human_review=args.review or args.action == "resume")
        if args.stream:
            # subgraphs=True 将研究子图的命名空间事件也写入流，便于观察嵌套执行。
            # subgraphs=True includes namespaced events from research subgraphs in the stream.
            for event in graph.stream(
                payload, config, context=context, stream_mode=args.stream, subgraphs=True
            ):
                dump(event)
            result = graph.get_state(config).values
        else:
            result = graph.invoke(payload, config, context=context)
        latest = graph.get_state(config)
        if any(task.interrupts for task in latest.tasks):
            print(f"已暂停。运行：lg-demo resume --db {args.db} --thread {thread}")
        else:
            print(result.get("report", ""))
            if result.get("approved") is False:
                print("审核结果：拒绝")


if __name__ == "__main__":
    main()
