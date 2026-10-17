"""
Ingest a repo from a local path or a public GitHub URL.

Usage:
    python scripts/ingest.py /path/to/repo my-repo-name
    python scripts/ingest.py https://github.com/owner/repo [repo-name]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest.pipeline import ingest_github, ingest_repo


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].startswith("https://") and len(args) in (1, 2):
        ingest_github(args[0], args[1] if len(args) == 2 else None)
    elif len(args) == 2:
        ingest_repo(Path(args[0]).resolve(), args[1])
    else:
        print(__doc__)
        sys.exit(1)
