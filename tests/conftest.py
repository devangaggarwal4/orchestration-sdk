from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_text():
    """Read a wire-format fixture. These are hand-written to match Ollama's
    api/types.go and Anthropic's documented SSE events, not live recordings."""

    def read(name: str) -> str:
        return (FIXTURES / name).read_text()

    return read
