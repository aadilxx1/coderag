"""
Ingest a repo from a local path or a public GitHub URL.

Incremental by default: only new or changed files are re-embedded. --full
re-embeds everything.

Usage:
    python scripts/ingest.py [--full] /path/to/repo my-repo-name
    python scripts/ingest.py [--full] https://github.com/owner/repo [repo-name]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest.pipeline import ingest_github, ingest_repo


if __name__ == "__main__":
    args = sys.argv[1:]
    full = "--full" in args
    args = [a for a in args if a != "--full"]
    if args and args[0].startswith("https://") and len(args) in (1, 2):
        ingest_github(args[0], args[1] if len(args) == 2 else None, full=full)
    elif len(args) == 2:
        ingest_repo(Path(args[0]).resolve(), args[1], full=full)
    else:
        print(__doc__)
        sys.exit(1)
