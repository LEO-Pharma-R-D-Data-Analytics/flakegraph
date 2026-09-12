from __future__ import annotations

import types

import pytest

from kg_processor.adapters.embeddings import sentence_transformers as module
from kg_processor.adapters.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)
from kg_processor.ports.embeddings import EmbedOptions


def test_sentence_transformers_embeddings_return_empty_without_loading_model() -> None:
    def factory(*_args: object, **_kwargs: object) -> _FakeModel:
        raise AssertionError("empty embedding input should not load a model")

    provider = SentenceTransformersEmbeddingProvider(factory)

    assert provider.embed([], EmbedOptions(model="embed", dimension=3)) == []


def test_sentence_transformers_embeddings_batch_and_validate_dimension() -> None:
    calls: list[tuple[str, str | None]] = []

    def factory(model_name: str, device: str | None = None) -> _FakeModel:
        calls.append((model_name, device))
        return _FakeModel([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    provider = SentenceTransformersEmbeddingProvider(factory, device="cpu")

    vectors = provider.embed(
        ["Alice", "Acme"],
        EmbedOptions(model="sentence-transformers/all-MiniLM-L6-v2", dimension=3, batch_size=2),
    )

    assert calls == [("sentence-transformers/all-MiniLM-L6-v2", "cpu")]
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_sentence_transformers_embeddings_accept_numpy_like_output() -> None:
    provider = SentenceTransformersEmbeddingProvider(
        lambda *_args, **_kwargs: _FakeModel(_NumpyLike([[0.1, 0.2, 0.3]]))
    )

    vectors = provider.embed(["Alice"], EmbedOptions(model="embed", dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]


def test_sentence_transformers_embeddings_accept_flat_single_vector() -> None:
    provider = SentenceTransformersEmbeddingProvider(
        lambda *_args, **_kwargs: _FakeModel([0.1, 0.2, 0.3])
    )

    vectors = provider.embed(["Alice"], EmbedOptions(model="embed", dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]


def test_sentence_transformers_embeddings_reject_dimension_mismatch() -> None:
    provider = SentenceTransformersEmbeddingProvider(lambda *_args, **_kwargs: _FakeModel([[0.1]]))

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        provider.embed(["Alice"], EmbedOptions(model="embed", dimension=3))


def test_sentence_transformers_embeddings_reject_non_numeric_values() -> None:
    provider = SentenceTransformersEmbeddingProvider(
        lambda *_args, **_kwargs: _FakeModel([["not-a-number"]])
    )

    with pytest.raises(ValueError, match="non-finite vector value"):
        provider.embed(["Alice"], EmbedOptions(model="embed", dimension=1))


def test_sentence_transformers_embeddings_reject_unsupported_matrix() -> None:
    provider = SentenceTransformersEmbeddingProvider(lambda *_args, **_kwargs: _FakeModel(object()))

    with pytest.raises(ValueError, match="unsupported embedding matrix"):
        provider.embed(["Alice"], EmbedOptions(model="embed", dimension=1))


def test_sentence_transformers_load_factory_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def import_module(name: str) -> types.ModuleType:
        if name == "sentence_transformers":
            raise ImportError("missing")
        return types.ModuleType(name)

    monkeypatch.setattr(
        "kg_processor.adapters.embeddings.sentence_transformers.importlib.import_module",
        import_module,
    )

    with pytest.raises(RuntimeError, match="local-embeddings extra"):
        module._load_sentence_transformer_factory()


def test_sentence_transformers_load_factory_requires_callable_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = types.ModuleType("sentence_transformers")
    package.SentenceTransformer = None  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "kg_processor.adapters.embeddings.sentence_transformers.importlib.import_module",
        lambda _name: package,
    )

    with pytest.raises(RuntimeError, match="not callable"):
        module._load_sentence_transformer_factory()


class _FakeModel:
    def __init__(self, vectors: object) -> None:
        self.vectors = vectors

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object:
        assert sentences
        assert batch_size > 0
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return self.vectors


class _NumpyLike:
    def __init__(self, value: list[list[float]]) -> None:
        self.value = value

    def tolist(self) -> list[list[float]]:
        return self.value
