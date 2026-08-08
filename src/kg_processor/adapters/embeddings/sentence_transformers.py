"""SentenceTransformers embedding adapter for the local/open-source profile."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable
from typing import Any, Protocol, cast

from kg_processor.ports.embeddings import EmbedOptions


class SentenceTransformerModel(Protocol):
    """Small protocol for the optional SentenceTransformers model object."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object:
        """Encode sentences using the options required by this adapter."""
        ...


SentenceTransformerFactory = Callable[..., SentenceTransformerModel]


class SentenceTransformersEmbeddingProvider:
    """Lazy-loads local SentenceTransformers models and normalizes vectors."""

    def __init__(
        self,
        model_factory: SentenceTransformerFactory | None = None,
        device: str | None = None,
    ) -> None:
        self.model_factory = model_factory or _load_sentence_transformer_factory()
        self.device = device
        self.models: dict[str, SentenceTransformerModel] = {}

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Encode text locally and enforce the configured embedding dimension."""

        if not texts:
            return []
        model = self._model(options.model)
        encoded = model.encode(
            texts,
            batch_size=options.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = _normalize_matrix(encoded)
        for vector in vectors:
            if len(vector) != options.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {options.dimension}, got {len(vector)}"
                )
        return vectors

    def _model(self, model_name: str) -> SentenceTransformerModel:
        if model_name not in self.models:
            if self.device:
                self.models[model_name] = self.model_factory(model_name, device=self.device)
            else:
                self.models[model_name] = self.model_factory(model_name)
        return self.models[model_name]


def _load_sentence_transformer_factory() -> SentenceTransformerFactory:
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise RuntimeError(
            "sentence_transformers embeddings require the local-embeddings extra: "
            "uv sync --extra local-embeddings"
        ) from exc
    factory = getattr(module, "SentenceTransformer", None)
    if not callable(factory):
        raise RuntimeError("sentence_transformers.SentenceTransformer is not callable")
    return cast(SentenceTransformerFactory, factory)


def _normalize_matrix(value: object) -> list[list[float]]:
    if hasattr(value, "tolist"):
        return _normalize_matrix(cast(Any, value).tolist())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | bytearray | dict):
        rows = list(value)
        if not rows:
            return []
        if all(isinstance(item, int | float) for item in rows):
            return [_normalize_vector(rows)]
        return [_normalize_vector(row) for row in rows]
    raise ValueError("sentence_transformers returned an unsupported embedding matrix")


def _normalize_vector(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        return _normalize_vector(cast(Any, value).tolist())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | bytearray | dict):
        vector: list[float] = []
        for item in value:
            if not isinstance(item, int | float) or not math.isfinite(float(item)):
                raise ValueError("sentence_transformers returned a non-finite vector value")
            vector.append(float(item))
        return vector
    raise ValueError("sentence_transformers returned an unsupported embedding vector")
