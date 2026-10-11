import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Telemetry defaults to ./hybrid_router.sqlite; keep tests out of the repo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HYBRID_ROUTER_CONFIG", raising=False)
