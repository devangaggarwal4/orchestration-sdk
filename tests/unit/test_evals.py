import pytest

from hybrid_router.config import load_config
from hybrid_router.evals import Answer, load_tasks, run_check, run_checks, run_evals
from hybrid_router.evals.tools import run_tool

TASKS = load_tasks()


def reference_answer(task) -> Answer:
    if task.reference is not None:
        return Answer(task.reference)
    steps = task.reference_steps or []
    calls = [(s["tool"], s["input"]) for s in steps if "tool" in s]
    final = next((s["final"] for s in reversed(steps) if "final" in s), "")
    return Answer(final, calls)


def test_task_set_shape():
    assert len(TASKS) == 30
    by_diff = {d: sum(t.difficulty == d for t in TASKS) for d in ("easy", "medium", "hard")}
    assert by_diff == {"easy": 10, "medium": 12, "hard": 8}
    assert all(t.reference is not None or t.reference_steps for t in TASKS)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_reference_passes_and_a_wrong_answer_fails(task):
    ok, failures = run_checks(task.checks, reference_answer(task))
    assert ok, failures
    assert not run_checks(task.checks, Answer("I'm not sure."))[0]


def test_reference_tool_calls_are_valid():
    for task in TASKS:
        for step in task.reference_steps or []:
            if "tool" in step:
                out, is_error = run_tool(step["tool"], step["input"])
                assert not is_error, (task.id, out)


def test_check_types():
    assert run_check(
        {"type": "json_fields", "expected": {"xs": ["a", "B"]}},
        Answer('```json\n{"xs": ["b", "a"]}\n```'),
    )[0]
    assert not run_check({"type": "json_fields", "expected": {"n": 2}}, Answer('{"n": 3}'))[0]
    assert run_check(
        {
            "type": "sql_result",
            "setup": "CREATE TABLE t(x); INSERT INTO t VALUES(1);",
            "expected": [[1]],
        },
        Answer("```sql\nSELECT x FROM t\n```"),
    )[0]
    bad_sql = run_check({"type": "sql_result", "setup": "", "expected": []}, Answer("SELEC"))
    assert not bad_sql[0] and "sql error" in bad_sql[1]
    loops = run_check(
        {"type": "python_tests", "tests": "assert True", "timeout_s": 1},
        Answer("```python\nwhile True: pass\n```"),
    )
    assert not loops[0]
    with pytest.raises(ValueError, match="unknown check"):
        run_check({"type": "vibes"}, Answer("x"))


def test_tools_report_errors():
    assert run_tool("get_weather", {"city": "Atlantis"})[1]
    assert run_tool("nope", {})[1]
    assert run_tool("calculator", {"expression": "(2+3)*4"}) == ("20", False)
    assert run_tool("calculator", {"expression": "__import__('os')"})[1]


async def test_simulated_run_compares_strategies(tmp_path):
    cfg = load_config(environ={})
    seen = []
    report = await run_evals(
        cfg, simulate=True, db_path=tmp_path / "e.sqlite", on_result=seen.append
    )
    s = {x.strategy: x for x in report.summaries}
    assert len(seen) == 150 and set(s) == {
        "hybrid",
        "pre_route",
        "cascade",
        "all_local",
        "all_cloud",
    }
    assert s["all_local"].local_share == 1 and s["all_cloud"].local_share == 0
    assert s["all_cloud"].cost_usd > s["hybrid"].cost_usd > s["all_local"].cost_usd
    assert s["hybrid"].pass_rate > s["all_local"].pass_rate
    assert s["hybrid"].savings_usd > 0
    assert s["cascade"].waste_latency_ms > 0 and s["pre_route"].waste_latency_ms == 0
    assert s["hybrid"].label_source == "all_local run"
    # all_cloud only fails the task labelled as needing Opus.
    failed = [r.task_id for r in report.results if r.strategy == "all_cloud" and not r.passed]
    assert failed == ["code-expression-parser"]


async def test_budget_stops_cloud_spending(tmp_path):
    cfg = load_config(environ={})
    report = await run_evals(
        cfg,
        strategies=["all_cloud"],
        simulate=True,
        budget_usd=0.002,
        db_path=tmp_path / "e.sqlite",
    )
    assert report.budget_exhausted
    assert any(r.skipped for r in report.results)
