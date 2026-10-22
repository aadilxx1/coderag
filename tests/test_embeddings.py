"""The NaN fallback in embed_texts, using a fake model (no real weights loaded)."""
import numpy as np
import pytest

from app.core import embeddings


class FakeModel:
    """Returns NaN for texts containing "bad" unless encoding on CPU."""

    device = "mps"

    def __init__(self, cpu_also_bad: bool = False):
        self.cpu_also_bad = cpu_also_bad
        self.cpu_calls: list[list[str]] = []

    def to(self, device):
        self.device = device

    def encode(self, texts, batch_size, convert_to_numpy, device=None):
        if device is not None:
            self.device = device  # like the real model: encode(device=) moves it
        if device == "cpu":
            self.cpu_calls.append(list(texts))
        broken = device != "cpu" or self.cpu_also_bad
        return np.array([[np.nan, np.nan] if broken and "bad" in t else [1.0, 2.0] for t in texts])


def test_non_finite_rows_are_recomputed_on_cpu(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embeddings, "_get_model", lambda: model)
    vectors = embeddings.embed_texts(["ok", "bad", "ok"])
    assert vectors == [[1.0, 2.0]] * 3
    assert model.cpu_calls == [["bad"]]  # only the bad row is redone
    assert model.device == "mps"         # and the model is moved back to the GPU


def test_clean_batch_never_falls_back(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embeddings, "_get_model", lambda: model)
    embeddings.embed_texts(["ok", "fine"])
    assert model.cpu_calls == []


def test_raises_if_cpu_is_also_non_finite(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel(cpu_also_bad=True))
    with pytest.raises(ValueError, match="non-finite"):
        embeddings.embed_texts(["bad"])
