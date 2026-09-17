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
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("模拟一次可恢复网络故障")
        return {"number": state["number"] * 2}

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
    assert len(attempts) == 2  # 第一次重试；第二次 invoke 命中缓存。
    return {"attempts": len(attempts), "second_call": "cached"}


def functional_lab():
    @task
    def square(value: int):
        return value * value

    @entrypoint(checkpointer=InMemorySaver())
    def workflow(values: list[int]):
        futures = [square(value) for value in values]
        return sum(future.result() for future in futures)

    return workflow.invoke([1, 2, 3], {"configurable": {"thread_id": "functional"}})


async def async_memory_lab():
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
