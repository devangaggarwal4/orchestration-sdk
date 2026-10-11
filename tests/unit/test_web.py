import json

import httpx2
import pytest
import yaml

from hybrid_router.config import load_config
from hybrid_router.router import HybridRouter
from hybrid_router.web.app import create_app, dump_config_yaml
from hybrid_router.web.demo import demo_backends


def make_app(tmp_path, *, token=None, static_dir=None, fake=True):
    cfg = load_config(
        environ={
            "HYBRID_ROUTER__TELEMETRY__DB_PATH": str(tmp_path / "t.sqlite"),
            "HYBRID_ROUTER__WEB__EVAL_DB_PATH": str(tmp_path / "e.sqlite"),
            "HYBRID_ROUTER__AGENT__WORKSPACE": str(tmp_path),
        }
    )
    (tmp_path / "notes.py").write_text("def escalate():\n    return 'opus'\n")
    router = HybridRouter(
        cfg, backends=demo_backends({t: c.model for t, c in cfg.tiers.items()}, speed=200)
    )
    return create_app(
        tmp_path / "hybrid_router.yaml",
        fake=fake,
        token=token,
        router=router,
        static_dir=static_dir,
    )


@pytest.fixture
async def client(tmp_path):
    app = make_app(tmp_path)
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def sse_events(client, method, url, **kw) -> list[dict]:
    events = []
    async with client.stream(method, url, **kw) as resp:
        assert resp.status_code == 200, await resp.aread()
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


async def test_auth_required_when_token_set(tmp_path):
    app = make_app(tmp_path, token="s3cret")
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.get("/api/meta")).json()["auth_required"] is True
        assert (await c.get("/api/conversations")).status_code == 401
        bad = await c.get("/api/conversations", headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401
        ok = await c.get("/api/conversations", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200


async def test_health_and_budget(client):
    h = (await client.get("/api/health")).json()
    assert [t["tier"] for t in h["tiers"]] == ["local", "sonnet", "opus"]
    assert all(t["ok"] for t in h["tiers"]) and h["fake"] is True
    assert (await client.get("/api/budget")).json()["cloud_allowed"] is True


async def test_chat_streams_persists_and_keeps_session(client):
    conv = (await client.post("/api/conversations", json={})).json()
    url = f"/api/conversations/{conv['id']}/messages"
    events = await sse_events(client, "POST", url, json={"content": "hi there, quick question"})
    types = [e["type"] for e in events]
    assert types[0] == "step_start" and "step_end" in types and types[-1] == "saved"
    deltas = [
        e["event"]["text"]
        for e in events
        if e["type"] == "router" and e["event"]["type"] == "text_delta"
    ]
    assert "".join(deltas).startswith("Quick answer")
    step_end = next(e for e in events if e["type"] == "step_end")
    assert step_end["decision"]["final_tier"] == "local" and "cost_usd" in step_end

    # A hard follow-up escalates (demo local model hedges on hard words).
    events = await sse_events(
        client, "POST", url, json={"content": "design the architecture and its trade-offs"}
    )
    d = next(e for e in events if e["type"] == "step_end")["decision"]
    assert d["final_tier"] in ("sonnet", "opus")
    assert d["session_id"] == f"chat-{conv['id']}" and d["step_index"] == 1

    saved = (await client.get(f"/api/conversations/{conv['id']}")).json()
    assert saved["title"] == "hi there, quick question"
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant", "user", "assistant"]
    assert saved["messages"][1]["decisions"][0]["final_tier"] == "local"
    assert (await client.delete(f"/api/conversations/{conv['id']}")).json()["deleted"]
    assert (await client.get(f"/api/conversations/{conv['id']}")).status_code == 404


async def test_chat_with_tools_runs_agent_steps(client):
    conv = (await client.post("/api/conversations", json={})).json()
    events = await sse_events(
        client,
        "POST",
        f"/api/conversations/{conv['id']}/messages",
        json={"content": "where is escalate defined?", "tools": True},
    )
    types = [e["type"] for e in events]
    assert "tool_call" in types and "tool_result" in types
    result = next(e for e in events if e["type"] == "tool_result")
    assert "notes.py:1" in result["content"]
    end = next(e for e in events if e["type"] == "agent_end")
    assert end["status"] == "done" and end["steps"] == 2


async def test_agent_run_lifecycle(client):
    run_id = (
        await client.post("/api/runs", json={"kind": "agent", "task": "where is escalate?"})
    ).json()["run_id"]
    events = await sse_events(client, "GET", f"/api/runs/{run_id}/events")
    assert events[-1]["type"] == "run_end" and events[-1]["status"] == "done"
    assert events[-1]["summary"]["steps"] == 2
    assert [e["seq"] for e in events] == list(range(len(events)))
    run = (await client.get(f"/api/runs/{run_id}")).json()
    assert run["status"] == "done" and run["kind"] == "agent"
    # Replaying a finished run gives the same events.
    assert await sse_events(client, "GET", f"/api/runs/{run_id}/events") == events
    assert (await client.post("/api/runs", json={"kind": "agent"})).status_code == 422


async def test_compare_prompt_and_task(client):
    run_id = (
        await client.post(
            "/api/runs",
            json={
                "kind": "compare",
                "prompt": "summarize: the cat sat",
                "strategies": ["hybrid", "all_local", "all_cloud"],
            },
        )
    ).json()["run_id"]
    events = await sse_events(client, "GET", f"/api/runs/{run_id}/events")
    legs = {e["leg"]: e for e in events if e["type"] == "leg_end"}
    assert set(legs) == {"hybrid", "all_local", "all_cloud"}
    assert legs["all_local"]["tiers"] == ["local"] and legs["all_cloud"]["tiers"] == ["sonnet"]
    assert legs["all_cloud"]["cost_usd"] > 0 == legs["all_local"]["cost_usd"]

    tasks = (await client.get("/api/eval/tasks")).json()
    assert len(tasks) == 30
    run_id = (
        await client.post(
            "/api/runs",
            json={
                "kind": "compare",
                "task_id": "arithmetic",
                "strategies": ["hybrid"],
            },
        )
    ).json()["run_id"]
    events = await sse_events(client, "GET", f"/api/runs/{run_id}/events")
    leg = next(e for e in events if e["type"] == "leg_end")
    assert "passed" in leg  # demo models don't know the answer; the check still runs
    bad = await client.post(
        "/api/runs", json={"kind": "compare", "prompt": "x", "strategies": ["yolo"]}
    )
    assert bad.status_code == 422


async def test_eval_run_simulated(client):
    run_id = (
        await client.post(
            "/api/runs",
            json={
                "kind": "eval",
                "strategies": ["hybrid", "all_local"],
                "only": ["arithmetic", "knights-knaves"],
            },
        )
    ).json()["run_id"]
    events = await sse_events(client, "GET", f"/api/runs/{run_id}/events")
    results = [e for e in events if e["type"] == "eval_result"]
    assert len(results) == 4
    summary = next(e for e in events if e["type"] == "eval_summary")
    assert {s["strategy"] for s in summary["summaries"]} == {"hybrid", "all_local"}


async def test_config_validate_whatif_apply(tmp_path):
    app = make_app(tmp_path, fake=False)
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://test"
    ) as c:
        cfg = (await c.get("/api/config")).json()
        assert cfg["exists"] is False and "tiers:" in cfg["yaml"]

        v = await c.post(
            "/api/config/validate", json={"patch": {"routing": {"hard_threshold": 0.5}}}
        )
        assert v.status_code == 200 and "+  hard_threshold: 0.5" in v.json()["diff"]
        bad = await c.post(
            "/api/config/validate", json={"patch": {"routing": {"easy_threshold": 0.9}}}
        )
        assert bad.status_code == 422 and "easy <= hard" in bad.text

        w = await c.post("/api/config/whatif", json={"patch": {"routing": {"hard_threshold": 0.3}}})
        assert w.status_code == 200 and "routes_considered" in w.json()

        put = (
            await c.put("/api/config", json={"patch": {"routing": {"hard_threshold": 0.5}}})
        ).json()
        assert put["applied"] and put["written"]
        written = yaml.safe_load((tmp_path / "hybrid_router.yaml").read_text())
        assert written["routing"]["hard_threshold"] == 0.5
        assert written["tiers"]["local"]["kind"] == "ollama"
        assert app.state.hr.router.config.routing.hard_threshold == 0.5
        # Second write keeps a backup of the first.
        put2 = (
            await c.put("/api/config", json={"patch": {"routing": {"easy_threshold": 0.2}}})
        ).json()
        assert (
            put2["backup"]
            and load_config(tmp_path / "hybrid_router.yaml", environ={}).routing.easy_threshold
            == 0.2
        )


def test_dumped_yaml_round_trips():
    cfg = load_config(
        environ={
            "HYBRID_ROUTER__ROUTING__HARD_THRESHOLD": "0.6",
            "HYBRID_ROUTER__TIERS__LOCAL__MODEL": "llama3.1:8b",
        }
    )
    from hybrid_router.web.app import load_config_from_dict

    again = load_config_from_dict(yaml.safe_load(dump_config_yaml(cfg)))
    assert again == cfg


async def test_stats_and_tune_endpoints(client):
    conv = (await client.post("/api/conversations", json={})).json()
    await sse_events(
        client, "POST", f"/api/conversations/{conv['id']}/messages", json={"content": "hello"}
    )
    stats = (await client.get("/api/stats", params={"since": "1d"})).json()
    assert stats["overview"]["routes"] >= 1 and "budget" in stats
    assert (await client.get("/api/stats", params={"since": "whenever"})).status_code == 422
    eval_stats = (await client.get("/api/stats", params={"source": "eval"})).json()
    assert eval_stats["overview"]["routes"] == 0
    tune = (await client.get("/api/tune")).json()
    assert tune["enough_data"] is False


async def test_spa_fallback_and_api_404(tmp_path):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>app</html>")
    (static / "assets" / "a.js").write_text("x")
    app = make_app(tmp_path, static_dir=static)
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.get("/dashboard")).text == "<html>app</html>"
        assert (await c.get("/assets/a.js")).text == "x"
        assert (await c.get("/api/nope")).status_code == 404
        assert (await c.get("/../../etc/passwd")).text == "<html>app</html>"
