import pytest

from app.core.retrieve import _is_test_file


@pytest.mark.parametrize("path", [
    "tests/test_chunker.py",
    "tests/test_itsdangerous/test_signer.py",
    "pkg/test/helpers.py",
    "test_foo.py",
    "pkg/foo_test.py",
    "tests/conftest.py",
])
def test_detects_test_files(path):
    assert _is_test_file(path)


@pytest.mark.parametrize("path", [
    "app/core/retrieve.py",
    "src/itsdangerous/signer.py",
    "pkg/testing_utils.py",
    "pkg/contest.py",
])
def test_ignores_source_files(path):
    assert not _is_test_file(path)
