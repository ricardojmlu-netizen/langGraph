"""Run: python examples/advanced.py. No API keys required."""

import asyncio
from typing import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import CachePolicy, RetryPolicy

from langgraph_demo.graph import Context, build_graph, initial_state


class Counter(TypedDict):
    number: int


def resilience_lab():
    attempts = []

    def flaky(state: Counter):
        # 故意让第一次调用失败，以验证 RetryPolicy 只重试指定的可恢复异常。
        # Intentionally fail once to show RetryPolicy only retries the chosen recoverable exception.
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("模拟一次可恢复网络故障")
        return {"number": state["number"] * 2}

    # cache 与 retry 都绑定在节点层面：先完成重试，后续相同输入才命中缓存。
    # Cache and retry live at node level: retries finish first, then identical calls hit cache.
    builder = StateGraph(Counter)
    builder.add_node(
        "flaky",
        flaky,
        retry_policy=RetryPolicy(
            max_attempts=3, initial_interval=0.01, jitter=False, retry_on=ConnectionError
        ),
        cache_policy=CachePolicy(ttl=60),
    )
    builder.add_edge(START, "flaky")
    builder.add_edge("flaky", END)
    graph = builder.compile(cache=InMemoryCache())
    assert graph.invoke({"number": 21})["number"] == 42
    assert graph.invoke({"number": 21})["number"] == 42
    # 第一次 invoke 包含一次失败和一次重试；第二次 invoke 命中缓存。
    # First invoke fails once then retries; the second invoke hits the cache.
    assert len(attempts) == 2
    return {"attempts": len(attempts), "second_call": "cached"}


def functional_lab():
    @task
    def square(value: int):
        # @task 把可独立调度的工作划为持久化执行边界。
        # @task marks independently schedulable work as a durable execution boundary.
        return value * value

    @entrypoint(checkpointer=InMemorySaver())
    def workflow(values: list[int]):
        # 先创建所有 future 再读取 result，让任务可并行调度。
        # Create all futures before reading results so tasks can be scheduled concurrently.
        futures = [square(value) for value in values]
        return sum(future.result() for future in futures)

    return workflow.invoke([1, 2, 3], {"configurable": {"thread_id": "functional"}})


async def async_memory_lab():
    # 同一 user_id 的 Store namespace 跨 thread 复用偏好；不同用户自然隔离。
    # The Store namespace reuses preferences across threads for one user_id and isolates other users.
    store = InMemoryStore()
    graph = build_graph(InMemorySaver(), store)
    first = await graph.ainvoke(
        initial_state("学习架构", ["architecture"]),
        {"configurable": {"thread_id": "a"}},
        context=Context(user_id="alice", preference="详细中文"),
    )
    second = await graph.ainvoke(
        initial_state("学习恢复", ["reliability"]),
        {"configurable": {"thread_id": "b"}},
        context=Context(user_id="alice", preference="其他偏好"),
    )
    assert first["preference"] == second["preference"] == "详细中文"
    events = []
    # astream 让异步调用方逐步消费图更新，而不是等待整个报告完成。
    # astream lets async callers consume graph updates incrementally instead of waiting for the report.
    async for event in graph.astream(
        initial_state("流式输出", ["operations"]),
        {"configurable": {"thread_id": "c"}},
        context=Context(),
        stream_mode="updates",
    ):
        events.append(event)
    return {"cross_thread_preference": second["preference"], "async_events": len(events)}


if __name__ == "__main__":
    print("Retry / cache:", resilience_lab())
    print("Functional API:", functional_lab())
    print("Async / Store:", asyncio.run(async_memory_lab()))
