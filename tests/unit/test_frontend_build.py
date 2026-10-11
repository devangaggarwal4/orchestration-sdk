import os
import sys
import time

import httpx2

from hybrid_router.web import frontend_build as fb
from hybrid_router.web.app import create_app


def make_frontend(tmp_path):
    fe = tmp_path / "frontend"
    (fe / "src").mkdir(parents=True)
    (fe / "package.json").write_text("{}")
    (fe / "src" / "main.tsx").write_text("x")
    return fe


def touch(path, ts):
    os.utime(path, (ts, ts))


def test_ui_status(tmp_path):
    fe = make_frontend(tmp_path)
    static = tmp_path / "static"
    assert fb.ui_status(static, fe) == "missing"
    assert fb.ui_status(static, tmp_path / "nope") == "no_source"

    static.mkdir()
    (static / "index.html").write_text("<html>")
    now = time.time()
    touch(fe / "src" / "main.tsx", now - 100)
    touch(fe / "package.json", now - 100)
    touch(static / "index.html", now - 50)
    assert fb.ui_status(static, fe) == "ok"

    touch(fe / "src" / "main.tsx", now)  # e.g. after git pull
    assert fb.ui_status(static, fe) == "stale"
    assert fb.ui_status(static, tmp_path / "nope") == "ok"  # prebuilt, no source


def test_build_ui_runs_install_then_build(tmp_path, monkeypatch):
    fe = make_frontend(tmp_path)
    log = tmp_path / "calls.txt"
    fake = tmp_path / "pnpm.py"
    fake.write_text(
        "import sys\n"
        f"open({str(log)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "sys.exit(1 if 'build' in sys.argv and 'FAIL' in open('package.json').read() else 0)\n"
    )
    monkeypatch.setattr(fb, "node_problem", lambda: None)
    monkeypatch.setattr(fb, "find_pnpm", lambda: [sys.executable, str(fake)])
    messages = []
    assert fb.build_ui(fe, echo=messages.append)
    assert log.read_text().splitlines() == ["install --frozen-lockfile", "build"]

    (fe / "package.json").write_text('{"FAIL": 1}')
    assert not fb.build_ui(fe, echo=messages.append)
    assert "failed (exit code 1)" in messages[-1]


def test_build_ui_explains_missing_tools(tmp_path, monkeypatch):
    msgs = []
    monkeypatch.setattr(fb, "node_problem", lambda: "Node.js is not installed. Install it with: x")
    assert not fb.build_ui(tmp_path, echo=msgs.append)
    assert "Node.js is not installed" in msgs[0]

    monkeypatch.setattr(fb, "node_problem", lambda: None)
    monkeypatch.setattr(fb, "find_pnpm", lambda: None)
    assert not fb.build_ui(tmp_path, echo=msgs.append)
    assert "pnpm is not available" in msgs[-1]


async def test_server_picks_up_a_build_without_restart(tmp_path):
    static = tmp_path / "static"
    app = create_app(tmp_path / "cfg.yaml", fake=True, static_dir=static)
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://test"
    ) as c:
        first = await c.get("/chat")
        assert first.status_code == 503 and "hasn't been built" in first.text
        assert (await c.get("/api/meta")).status_code == 200  # API works regardless

        (static / "assets").mkdir(parents=True)
        (static / "index.html").write_text("<html>app</html>")
        (static / "assets" / "a.js").write_text("js")
        assert (await c.get("/chat")).text == "<html>app</html>"
        assert (await c.get("/assets/a.js")).text == "js"
