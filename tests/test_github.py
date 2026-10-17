import pytest

from app.ingest.github import parse_github_url


@pytest.mark.parametrize("url", [
    "https://github.com/tiangolo/fastapi",
    "https://github.com/tiangolo/fastapi/",
    "https://github.com/tiangolo/fastapi.git",
])
def test_parses_plain_urls(url):
    repo = parse_github_url(url)
    assert repo.slug == "tiangolo/fastapi"
    assert repo.clone_url == "https://github.com/tiangolo/fastapi.git"
    assert repo.ref is None


def test_parses_branch_from_tree_url():
    assert parse_github_url("https://github.com/psf/requests/tree/v2.31.0").ref == "v2.31.0"


def test_explicit_ref_overrides_url():
    assert parse_github_url("https://github.com/psf/requests/tree/main", ref="dev").ref == "dev"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "/Users/someone/repo",
    "http://github.com/owner/repo",          # not https
    "https://gitlab.com/owner/repo",
    "https://github.com.evil.com/owner/repo",
    "https://github.com/owner",              # no repo
    "https://github.com/owner/repo/issues",
    "https://github.com/owner/..",
])
def test_rejects_non_github_urls(url):
    with pytest.raises(ValueError):
        parse_github_url(url)


def test_rejects_ref_that_looks_like_git_option():
    with pytest.raises(ValueError):
        parse_github_url("https://github.com/owner/repo", ref="--upload-pack=evil")
