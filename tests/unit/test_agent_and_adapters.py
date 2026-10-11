import pytest

from hybrid_router.adapters.claude_agent_sdk import delegate_local
from hybrid_router.agent import (
    Agent,
    AgentEndEvent,
    StepEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    workspace_tools,
)
from hybrid_router.config import load_config
from hybrid_router.reporting.queries import RouteRow
from hybrid_router.router import HybridRouter, build_scorer
from hybrid_router.scoring import RuleScorer, ScoringContext
from hybrid_router.scoring.embedding import EmbeddingIndex, EmbeddingScorer, cosine
from hybrid_router.scoring.learned import LearnedScorer, train
from hybrid_router.testing import Reply, ScriptedBackend
from hybrid_router.types import CompletionRequest, Message, Tier

GOOD = "A complete, confident and correct answer to the question."


def router_with(local, cloud=None):
    backends = {
        Tier.LOCAL: ScriptedBackend(local, tier=Tier.LOCAL),
        Tier.SONNET: ScriptedBackend(cloud or [GOOD], tier=Tier.SONNET),
        Tier.OPUS: ScriptedBackend(cloud or [GOOD], tier=Tier.OPUS),
    }
    return HybridRouter(load_config(environ={}), backends=backends, telemetry=None), backends


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "app.py").write_text("def handler():\n    return 42\n")
    (tmp_path / "notes.md").write_text("handler returns the answer\n")
    (tmp_path / ".secret").write_text("token")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.cpython-313.pyc").write_bytes(b"\0handler\0")
    (tmp_path / "blob.bin").write_bytes(b"handler\0\x01")
    return tmp_path


def tools_by_name(ws, **kw):
    return {t.spec.name: t for t in workspace_tools(ws, **kw)}


# --- tools -------------------------------------------------------------------------


async def test_read_and_search_are_confined_to_workspace(workspace):
    from hybrid_router.agent import ToolError

    t = tools_by_name(workspace)
    assert "return 42" in await t["read_file"].fn({"path": "app.py"})
    with pytest.raises(ToolError, match="outside the workspace"):
        await t["read_file"].fn({"path": "../../etc/passwd"})
    hits = await t["search_text"].fn({"pattern": "handler", "glob": "*.py"})
    assert hits == "app.py:1: def handler():"
    assert "token" not in await t["search_text"].fn({"pattern": "token"})  # dotfiles skipped
    assert "run_python" not in t


async def test_run_python_only_when_allowed(workspace):
    from hybrid_router.agent import ToolError

    t = tools_by_name(workspace, allow_code_exec=True, exec_timeout_s=2)
    assert (await t["run_python"].fn({"code": "print(6*7)"})).strip() == "42"
    with pytest.raises(ToolError, match="exit code"):
        await t["run_python"].fn({"code": "raise SystemExit(3)"})
    with pytest.raises(ToolError, match="timed out"):
        await t["run_python"].fn({"code": "while True: pass"})


# --- agent loop --------------------------------------------------------------------


async def test_agent_runs_tools_and_routes_every_step(workspace):
    router, backends = router_with(
        [
            Reply(tool_calls=[("search_text", {"pattern": "handler"})]),
            Reply(tool_calls=[("read_file", {"path": "missing.py"})]),
            Reply(GOOD),
        ]
    )
    agent = Agent(router, workspace_tools(workspace), session_id="s1")
    events = [e async for e in agent.run("what does handler return?")]
    steps = [e for e in events if isinstance(e, StepEndEvent)]
    assert [s.decision.step_index for s in steps] == [0, 1, 2]
    assert all(s.decision.session_id == "s1" for s in steps)
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert not results[0].is_error and results[1].is_error  # missing file
    end = events[-1]
    assert isinstance(end, AgentEndEvent) and end.status == "done" and end.answer == GOOD
    # The failed local tool call raised the next step's score (tool-error feedback).
    assert "previous local step's tool call errored" in " ".join(steps[2].decision.reasons)


async def test_agent_unknown_tool_and_max_steps(workspace):
    router, _ = router_with([Reply(tool_calls=[("read_file", {"path": "app.py"})])])
    agent = Agent(router, workspace_tools(workspace), max_steps=2)
    events = [e async for e in agent.run("loop forever")]
    assert isinstance(events[-1], AgentEndEvent) and events[-1].status == "max_steps"
    assert sum(isinstance(e, ToolCallEvent) for e in events) == 2


# --- LangGraph adapter ---------------------------------------------------------------


async def test_langgraph_chat_model_round_trip():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool

    from hybrid_router.adapters.langgraph import HybridRouterChatModel, to_router_messages

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    router, backends = router_with([Reply(tool_calls=[("get_weather", {"city": "Oslo"})]), GOOD])
    llm = HybridRouterChatModel(router=router, task_type="tool_use").bind_tools([get_weather])
    first = await llm.ainvoke([SystemMessage("be brief"), HumanMessage("weather in Oslo?")])
    assert isinstance(first, AIMessage)
    assert first.tool_calls[0]["name"] == "get_weather"
    assert first.response_metadata["route"]["final_tier"] == "local"
    sent = backends[Tier.LOCAL].requests[0]
    assert sent.system == "be brief" and sent.tools[0].name == "get_weather"

    history = [
        HumanMessage("weather in Oslo?"),
        first,
        ToolMessage("sunny", tool_call_id=first.tool_calls[0]["id"]),
    ]
    system, msgs = to_router_messages(history)
    assert system is None and [m.role for m in msgs] == ["user", "assistant", "user"]
    second = await llm.ainvoke(history)
    assert second.content == GOOD


# --- Claude Agent SDK tool --------------------------------------------------------------


async def test_delegate_local_handler():
    router, _ = router_with([GOOD])
    out = await delegate_local(router, {"task": "classify: great movie", "task_type": "classify"})
    text = out["content"][0]["text"]
    assert text.startswith(GOOD) and "served by local" in text


def test_delegate_local_server_builds_when_sdk_installed():
    pytest.importorskip("claude_agent_sdk")
    from hybrid_router.adapters.claude_agent_sdk import delegate_local_server

    router, _ = router_with([GOOD])
    assert delegate_local_server(router) is not None


# --- learned & embedding scorers ------------------------------------------------------------


def route(task_type_feature: float, failed: bool) -> RouteRow:
    return RouteRow(
        route_id="r",
        ts=0,
        session_id=None,
        task_type="t",
        strategy="hybrid",
        score=0.5,
        zone="uncertain",
        features={
            "tokens": 0,
            "tools": 0,
            "task_type": task_type_feature,
            "keywords": 0,
            "schema": 0,
        },
        forced=False,
        planned_tier="local",
        final_tier="local",
        escalated=failed,
        escalation_reason=None,
        success=True,
        total_latency_ms=1,
        actual_cost_usd=0,
        counterfactual_cost_usd=0,
        config_hash="h",
        local_role="cascade_attempt",
        local_ok=not failed,
        local_succeeded=not failed,
        local_latency_ms=1,
        local_failed_validator=None,
        cloud_cost_usd=0,
        cloud_latency_ms=None,
        waste_latency_ms=0,
    )


def test_learned_scorer_trains_and_scores(tmp_path):
    routes = [route(0.1, False)] * 40 + [route(0.9, True)] * 40
    model = train(routes)
    assert model.auc == 1.0 and model.weights[2] > 0  # task_type drives failure
    path = tmp_path / "m.json"
    model.save(path)
    scorer = LearnedScorer.load(path)
    hard = scorer.score_features({"task_type": 0.9})
    easy = scorer.score_features({"task_type": 0.1})
    assert hard.score > 0.8 > 0.2 > easy.score
    assert hard.reasons[0].startswith("learned model")
    with pytest.raises(ValueError, match="need 50"):
        train(routes[:10])

    cfg = load_config(
        environ={
            "HYBRID_ROUTER__SCORER__KIND": "learned",
            "HYBRID_ROUTER__SCORER__LEARNED_MODEL_PATH": str(path),
        }
    )
    assert isinstance(build_scorer(cfg), LearnedScorer)
    missing = load_config(
        environ={
            "HYBRID_ROUTER__SCORER__KIND": "learned",
            "HYBRID_ROUTER__SCORER__LEARNED_MODEL_PATH": "nope.json",
        }
    )
    assert isinstance(build_scorer(missing), RuleScorer)


async def test_embedding_scorer_knn():
    class FakeEmbedder:
        async def embed(self, texts):
            return [[1.0, 0.0] if "hard" in t else [0.0, 1.0] for t in texts]

    index = EmbeddingIndex()
    for _ in range(30):
        index.add([1.0, 0.1], True)
        index.add([0.1, 1.0], False)
    assert cosine([1, 0], [1, 0]) == pytest.approx(1)
    scorer = EmbeddingScorer(index, FakeEmbedder(), RuleScorer(), k=10)  # type: ignore[arg-type]
    req = CompletionRequest(messages=[Message(role="user", content="a hard one")])
    hard = await scorer.ascore(req, ScoringContext())
    assert hard.score == 1.0 and "10/10" in hard.reasons[0]
    easy = await scorer.ascore(
        CompletionRequest(messages=[Message(role="user", content="easy")]), ScoringContext()
    )
    assert easy.score == 0.0
    # Sync path falls back to the rule scorer without a precomputed embedding.
    assert scorer.score(req, ScoringContext()).reasons != hard.reasons
    small = EmbeddingScorer(EmbeddingIndex(), FakeEmbedder(), RuleScorer())  # type: ignore[arg-type]
    assert "too small" in (await small.ascore(req, ScoringContext())).reasons[0]
