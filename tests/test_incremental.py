"""Unit tests for incremental-ingest change detection (no database needed)."""
from pathlib import Path

from app.ingest import pipeline
from app.ingest.pipeline import file_hash, plan_changes


def test_classifies_added_modified_removed_unchanged():
    current = {"a.py": "h1", "b.py": "h2-new", "d.py": "h4"}
    stored = {"a.py": "h1", "b.py": "h2", "c.py": "h3"}
    plan = plan_changes(current, stored, chunk_paths={"a.py", "b.py", "c.py"})
    assert plan.unchanged == ["a.py"]
    assert plan.modified == ["b.py"]
    assert plan.added == ["d.py"]
    assert plan.removed == ["c.py"]
    assert plan.to_reindex == ["d.py", "b.py"]


def test_first_run_on_empty_db_adds_everything():
    plan = plan_changes({"a.py": "h1", "b.py": "h2"}, stored={}, chunk_paths=set())
    assert plan.added == ["a.py", "b.py"]
    assert not (plan.modified or plan.removed or plan.unchanged)


def test_repo_ingested_before_hash_tracking_is_fully_replaced():
    # Rows exist in code_chunks but no hashes were recorded: every file must be
    # re-indexed (as modified, so its old rows get deleted), and rows for files
    # that are gone must be removed.
    plan = plan_changes({"a.py": "h1"}, stored={}, chunk_paths={"a.py", "gone.py"})
    assert plan.modified == ["a.py"]
    assert plan.removed == ["gone.py"]
    assert not plan.added


def test_file_without_chunks_is_still_tracked():
    # e.g. an empty __init__.py: in ingested_files but not code_chunks.
    plan = plan_changes({"__init__.py": "h"}, stored={"__init__.py": "h"}, chunk_paths=set())
    assert plan.unchanged == ["__init__.py"]


def test_hash_tracks_content_and_index_version(tmp_path: Path, monkeypatch):
    f = tmp_path / "x.py"
    f.write_text("a = 1\n")
    h1 = file_hash(f)
    assert file_hash(f) == h1

    f.write_text("a = 2\n")
    assert file_hash(f) != h1

    h2 = file_hash(f)
    monkeypatch.setattr(pipeline, "INDEX_VERSION", "999")
    assert file_hash(f) != h2
