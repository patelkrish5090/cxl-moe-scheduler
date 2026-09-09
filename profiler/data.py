"""Corpus loading and sequence packing for the activation profiler.

WikiText-2 is the docs.md default. Documents are concatenated and cut into
fixed-length sequences ("packing") rather than padded individually, so every
position carries a real token and the activation histogram is not diluted by
padding.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterator

import numpy as np

from .config import DataConfig

# Datasets that used to live at a bare "canonical" id with no namespace. Recent
# huggingface_hub versions parse dataset paths as HF URIs and reject a repo id
# without a "namespace/name" form, so `load_dataset("wikitext", ...)` fails with
# HfUriError deep inside the resolver. These are the same datasets at their
# current namespaced locations.
_LEGACY_DATASET_IDS: dict[str, str] = {
    "wikitext": "Salesforce/wikitext",
    "ptb_text_only": "ptb-text-only/ptb_text_only",
    "c4": "allenai/c4",
}


def resolve_dataset_id(dataset: str) -> str:
    """Map a legacy bare dataset id onto its namespaced equivalent.

    Returns ``dataset`` unchanged when it already has a namespace or is unknown.
    """
    if "/" in dataset:
        return dataset
    replacement = _LEGACY_DATASET_IDS.get(dataset)
    if replacement is None:
        return dataset
    warnings.warn(
        f"dataset id {dataset!r} has no namespace; using {replacement!r} instead. "
        "Update the run config to silence this.",
        stacklevel=3,
    )
    return replacement


def load_token_stream(cfg: DataConfig, tokenizer: Any) -> np.ndarray:
    """Tokenise the configured corpus into one flat token-id array.

    Args:
        cfg: Dataset selection and packing parameters.
        tokenizer: A Hugging Face tokenizer.

    Returns:
        1-D int64 array of token ids, documents concatenated in order.

    Raises:
        RuntimeError: if ``datasets`` is not installed or the split is empty.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the `datasets` package is required to load a corpus; "
            "pip install -r requirements.txt"
        ) from exc

    dataset_id = resolve_dataset_id(cfg.dataset)
    try:
        dataset = (
            load_dataset(dataset_id, cfg.subset, split=cfg.split)
            if cfg.subset
            else load_dataset(dataset_id, split=cfg.split)
        )
    except Exception as exc:
        if "namespace/name" in str(exc) or "HfUriError" in type(exc).__name__:
            raise RuntimeError(
                f"the installed huggingface_hub rejects the dataset id {dataset_id!r} "
                "because it has no namespace. Set data.dataset in the run config to the "
                "namespaced form (for WikiText-2 that is 'Salesforce/wikitext')."
            ) from exc
        raise

    text_column = "text"
    if text_column not in dataset.column_names:
        candidates = [c for c in dataset.column_names if dataset.features[c].dtype == "string"]
        if not candidates:
            raise RuntimeError(
                f"dataset {cfg.dataset!r} has no string column to profile on; "
                f"columns are {dataset.column_names}"
            )
        text_column = candidates[0]

    texts = [t for t in dataset[text_column] if t and t.strip()]
    if not texts:
        raise RuntimeError(f"dataset {cfg.dataset}/{cfg.subset} split {cfg.split} is empty")

    if cfg.shuffle_documents:
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(texts)

    # Enough documents to fill max_sequences without tokenising the whole split.
    if cfg.max_sequences is not None:
        needed_chars = cfg.max_sequences * cfg.seq_len * 6  # ~6 chars/token, generous
        kept, total_chars = [], 0
        for text in texts:
            kept.append(text)
            total_chars += len(text)
            if total_chars >= needed_chars:
                break
        texts = kept

    encoded = tokenizer("\n\n".join(texts), return_attention_mask=False)["input_ids"]
    return np.asarray(encoded, dtype=np.int64)


def pack_sequences(token_ids: np.ndarray, seq_len: int, max_sequences: int | None) -> np.ndarray:
    """Cut a flat token stream into ``[n_sequences, seq_len]``.

    A trailing partial sequence is dropped so every row is full-length.

    Raises:
        ValueError: if the stream is shorter than one full sequence.
    """
    n_full = len(token_ids) // seq_len
    if n_full == 0:
        raise ValueError(
            f"corpus has {len(token_ids)} tokens, fewer than one sequence of {seq_len}; "
            "lower data.seq_len or widen the split"
        )
    if max_sequences is not None:
        n_full = min(n_full, max_sequences)
    return token_ids[: n_full * seq_len].reshape(n_full, seq_len)


def iter_batches(sequences: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield ``[<=batch_size, seq_len]`` blocks of packed sequences."""
    for start in range(0, sequences.shape[0], batch_size):
        yield sequences[start : start + batch_size]


def synthetic_token_stream(vocab_size: int, n_tokens: int, seed: int = 0) -> np.ndarray:
    """Random token ids, for plumbing tests when no corpus is available.

    Routing statistics from synthetic tokens are NOT evidence about real expert
    skew -- they only prove the hooks fire and the writers work.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=n_tokens, dtype=np.int64)
