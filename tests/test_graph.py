import asyncio
import subprocess
import sys

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from langgraph_demo.graph import Context, build_graph, initial_state


def config(thread="test"):
    return {"configurable": {"thread_id": thread}, "recursion_limit": 40}


def test_parallel_agent_and_tool_results():
    graph = build_graph(InMemorySaver(), InMemoryStore())
    result = graph.invoke(initial_state("设计 Agent"), config(), context=Context())
    assert len(result["findings"]) == 3
    assert {x["topic"] for x in result["findings"]} == {"architecture", "reliability", "operations"}
    assert "Checkpointer" in result["report"]
    assert result["approved"] is True
    assert len(result["messages"]) == 2
    assert not graph.get_state(config()).next


@pytest.mark.parametrize("approved", [True, False])
def test_interrupt_resume_after_reopen(tmp_path, approved):
    path = str(tmp_path / "checkpoint.sqlite")
    with SqliteSaver.from_conn_string(path) as saver:
        graph = build_graph(saver)
        graph.invoke(initial_state("审核报告"), config(), context=Context(human_review=True))
        assert any(t.interrupts for t in graph.get_state(config()).tasks)
    with SqliteSaver.from_conn_string(path) as saver:
        graph = build_graph(saver)
        result = graph.invoke(
            Command(resume={"approved": approved}), config(), context=Context(human_review=True)
        )
        assert result["approved"] is approved
        assert len(result["findings"]) == 3
        assert not graph.get_state(config()).next


def test_history_fork_preserves_original_checkpoint():
    graph = build_graph(InMemorySaver())
    graph.invoke(initial_state("原始报告"), config(), context=Context())
    old = next(x for x in graph.get_state_history(config()) if x.next == ("review",))
    branch = graph.update_state(old.config, {"report": "人工修订报告"}, as_node="synthesize")
    result = graph.invoke(None, branch, context=Context())
    assert result["report"] == "人工修订报告"
    assert graph.get_state(old.config).values["report"] != "人工修订报告"


def test_custom_stream_and_subgraph_events():
    graph = build_graph(InMemorySaver())
    events = list(
        graph.stream(
            initial_state("stream"),
            config(),
            context=Context(),
            stream_mode=["updates", "custom"],
            subgraphs=True,
        )
    )
    assert any(mode == "custom" and value.get("event") == "planned" for _, mode, value in events)
    assert any(namespace for namespace, _, _ in events)


def test_user_memory_isolation():
    store = InMemoryStore()
    graph = build_graph(InMemorySaver(), store)
    for thread, user, style in [
        ("a", "alice", "详细"),
        ("b", "alice", "简短"),
        ("c", "bob", "英文"),
    ]:
        result = graph.invoke(
            initial_state("记忆"), config(thread), context=Context(user_id=user, preference=style)
        )
        assert result["preference"] == ("详细" if user == "alice" else "英文")


def test_input_validation():
    with pytest.raises(ValueError):
        initial_state("bad", [])
    with pytest.raises(ValueError):
        initial_state("bad", ["unknown"])
    with pytest.raises(ValueError):
        build_graph().invoke(initial_state(" "), context=Context())


def test_advanced_labs():
    from examples.advanced import async_memory_lab, functional_lab, resilience_lab

    assert resilience_lab()["attempts"] == 2
    assert functional_lab() == 14
    assert asyncio.run(async_memory_lab())["async_events"] > 0


def test_cli_cross_process_resume(tmp_path):
    db = str(tmp_path / "cli.sqlite")
    base = [sys.executable, "-m", "langgraph_demo.cli"]
    first = subprocess.run(
        base + ["run", "--db", db, "--thread", "cli", "--review"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "已暂停" in first.stdout
    second = subprocess.run(
        base + ["resume", "--db", db, "--thread", "cli"], capture_output=True, text=True, check=True
    )
    assert "实施建议" in second.stdout
    duplicate = subprocess.run(
        base + ["run", "--db", db, "--thread", "cli"], capture_output=True, text=True, check=False
    )
    assert duplicate.returncode != 0


def test_deepseek_configuration(monkeypatch):
    from langgraph_demo.model import deepseek_model

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        deepseek_model()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-placeholder")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    model = deepseek_model()
    assert model.model_name == "deepseek-chat"
    assert model.openai_api_base == "https://api.deepseek.com"
