import asyncio
import json

from typer.testing import CliRunner

from hybrid_router.cli import app
from hybrid_router.config import load_config
from hybrid_router.evals import run_evals

runner = CliRunner()


def test_config_command():
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0 and "qwen3:8b" in result.output


def test_report_tune_whatif_on_simulated_db(tmp_path):
    db = tmp_path / "sim.sqlite"
    asyncio.run(
        run_evals(
            load_config(environ={}), strategies=["cascade"], simulate=True, repeat=2, db_path=db
        )
    )
    rep = runner.invoke(app, ["report", "--db", str(db), "--since", "1d", "--json"])
    assert rep.exit_code == 0, rep.output
    assert json.loads(rep.output)["overview"]["routes"] > 0

    tune = runner.invoke(app, ["tune", "--db", str(db), "--since", "1d"])
    assert tune.exit_code == 0 and "local attempt" in tune.output

    whatif = runner.invoke(app, ["whatif", "--db", str(db), "--set", "routing.hard_threshold=0.4"])
    assert whatif.exit_code == 0 and "would change" in whatif.output

    missing = runner.invoke(app, ["report", "--db", str(tmp_path / "nope.sqlite")])
    assert missing.exit_code == 1 and "no telemetry database" in missing.output


def test_eval_command_simulated(tmp_path):
    out = tmp_path / "r.json"
    result = runner.invoke(
        app,
        [
            "eval",
            "--simulate",
            "--strategies",
            "hybrid",
            "--only",
            "arithmetic,knights-knaves",
            "--db",
            str(tmp_path / "e.sqlite"),
            "--json",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["summaries"][0]["tasks"] == 2
