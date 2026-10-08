"""Build the local Chroma index from data/policies/*.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resolvebot.rag.ingest import ingest  # noqa: E402


def main() -> None:
    result = ingest(rebuild=True)
    print("Indexed policy files:", ", ".join(result["files"]))
    print("Chunks:", result["chunks"])
    print("Saved to:", result["persist_directory"])


if __name__ == "__main__":
    main()
