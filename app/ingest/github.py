"""
Fetch a public GitHub repo so it can be ingested.

Shallow-clones (--depth 1) into a temp dir with the system `git` -- we only
need the current snapshot of the code, not its history, and shelling out
avoids adding a GitPython dependency.

Only https://github.com/<owner>/<repo> URLs are accepted. Passing arbitrary
strings to `git clone` would let an API caller clone local paths (file://,
/etc/...) or hit internal hosts, so the URL is parsed and rebuilt rather
than passed through.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CLONE_TIMEOUT_S = 120

# owner: GitHub usernames are alphanumeric + hyphens.
# repo: alphanumeric plus . _ -  (a trailing .git is stripped separately).
_GITHUB_URL = re.compile(
    r"^https://github\.com/"
    r"(?P<owner>[A-Za-z0-9-]+)/"
    r"(?P<repo>[A-Za-z0-9._-]+?)"
    r"(?:\.git)?"
    r"(?:/tree/(?P<ref>[^\s]+?))?"
    r"/?$"
)


class GitHubCloneError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitHubRepo:
    owner: str
    name: str
    ref: str | None = None   # branch or tag; None = default branch

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"

    @property
    def slug(self) -> str:
        """Default repo name in code_chunks, e.g. 'tiangolo/fastapi'."""
        return f"{self.owner}/{self.name}"


def parse_github_url(url: str, ref: str | None = None) -> GitHubRepo:
    """Accepts https://github.com/owner/repo, with optional .git suffix or
    /tree/<branch>. An explicit `ref` overrides one found in the URL."""
    match = _GITHUB_URL.match(url.strip())
    if not match or match["repo"] in {".", ".."}:
        raise ValueError(f"Not a GitHub repo URL: {url!r} (expected https://github.com/owner/repo)")
    ref = ref or match["ref"]
    if ref is not None and (ref.startswith("-") or not re.fullmatch(r"[A-Za-z0-9._/-]+", ref)):
        # Leading '-' would be read by git as an option.
        raise ValueError(f"Invalid branch/tag name: {ref!r}")
    return GitHubRepo(owner=match["owner"], name=match["repo"], ref=ref)


@contextmanager
def cloned_repo(repo: GitHubRepo) -> Iterator[Path]:
    """Shallow-clone `repo` into a temp dir, yield its path, delete it after."""
    with tempfile.TemporaryDirectory(prefix="coderag-") as tmp:
        dest = Path(tmp) / repo.name
        cmd = ["git", "clone", "--depth", "1", "--quiet"]
        if repo.ref:
            cmd += ["--branch", repo.ref]
        cmd += ["--", repo.clone_url, str(dest)]

        # GIT_TERMINAL_PROMPT=0: a private or nonexistent repo makes git ask
        # for credentials, which would hang the request -- fail fast instead.
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                           timeout=CLONE_TIMEOUT_S, env=env)
        except subprocess.TimeoutExpired as e:
            raise GitHubCloneError(f"Cloning {repo.slug} timed out after {CLONE_TIMEOUT_S}s") from e
        except subprocess.CalledProcessError as e:
            detail = e.stderr.strip() or "unknown error"
            raise GitHubCloneError(
                f"Could not clone {repo.slug} (is it public, and does the branch exist?): {detail}"
            ) from e
        yield dest
