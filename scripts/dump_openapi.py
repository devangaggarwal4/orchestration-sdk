"""Write the web API's OpenAPI schema (used to generate frontend types)."""

import json
import sys
import tempfile
from pathlib import Path

from hybrid_router.web.app import create_app


def main(out: str) -> None:
    target = Path(out).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        import os

        os.chdir(tmp)  # keep the throwaway DB out of the repo
        app = create_app(Path(tmp) / "cfg.yaml", fake=True, static_dir=None)
        schema = app.openapi()
    target.write_text(json.dumps(schema, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "openapi.json")
