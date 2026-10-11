"""Build the web UI on demand, so `hybrid-router web` works without a separate step.

The built files live in ``web/static`` (git-ignored). When running from a source
checkout, the React source is in ``<repo>/frontend``; we rebuild when the build
is missing or older than any source file (e.g. after ``git pull``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal

STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

UiStatus = Literal["ok", "missing", "stale", "no_source"]

# Inputs whose changes require a rebuild.
_SOURCE_GLOBS = ("src/**/*", "index.html", "package.json", "pnpm-lock.yaml", "vite.config.ts")


def ui_status(static_dir: Path = STATIC_DIR, frontend_dir: Path = FRONTEND_DIR) -> UiStatus:
    index = static_dir / "index.html"
    has_source = (frontend_dir / "package.json").is_file()
    if not index.is_file():
        return "missing" if has_source else "no_source"
    if not has_source:
        return "ok"  # installed package with a prebuilt UI
    built_at = index.stat().st_mtime
    for pattern in _SOURCE_GLOBS:
        for f in frontend_dir.glob(pattern):
            if f.is_file() and f.stat().st_mtime > built_at:
                return "stale"
    return "ok"


def find_pnpm() -> list[str] | None:
    """The command to run pnpm: pnpm itself, else via corepack (bundled with Node)."""
    if pnpm := shutil.which("pnpm"):
        return [pnpm]
    if corepack := shutil.which("corepack"):
        return [corepack, "pnpm"]
    return None


def node_problem() -> str | None:
    """None if Node is usable, else a human explanation of what to install."""
    node = shutil.which("node")
    if node is None:
        return "Node.js is not installed. Install it with:  brew install node"
    try:
        out = subprocess.run(
            [node, "-p", "process.versions.node"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        major, minor = (int(x) for x in out.split(".")[:2])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None  # can't tell; let the build try
    if (major, minor) >= (22, 12) or (major == 20 and minor >= 19) or major > 22:
        return None
    return f"Node.js {out} is too old (need 20.19+ or 22.12+). Upgrade with:  brew upgrade node"


def build_ui(frontend_dir: Path = FRONTEND_DIR, *, echo=print) -> bool:
    """Install dependencies and build. Output streams to the terminal. True on success."""
    if problem := node_problem():
        echo(problem)
        return False
    pnpm = find_pnpm()
    if pnpm is None:
        echo("pnpm is not available. Install it with:  npm install -g pnpm@10")
        return False
    for args in (["install", "--frozen-lockfile"], ["build"]):
        cmd = [*pnpm, *args]
        echo(f"$ {' '.join(['pnpm', *args])}   (in {frontend_dir})")
        try:
            result = subprocess.run(cmd, cwd=frontend_dir)
        except OSError as e:
            echo(f"could not run pnpm: {e}")
            return False
        if result.returncode != 0:
            echo(f"`pnpm {' '.join(args)}` failed (exit code {result.returncode}); see above.")
            return False
    return True
