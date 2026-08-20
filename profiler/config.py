"""Configuration objects for the stage-1 activation profiler.

Configs are plain JSON on disk (no YAML dependency). See ``configs/*.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Which model to profile and how to place it.

    Attributes:
        name_or_path: Hugging Face repo id or local directory.
        dtype: Torch dtype name ("bfloat16", "float16", "float32", "auto").
        device_map: Passed to ``from_pretrained``. "auto" shards across visible
            GPUs; "cpu" forces CPU; None loads onto the default device.
        trust_remote_code: Only enable for repos you have vetted.
        attn_implementation: e.g. "sdpa", "eager", "flash_attention_2", or None
            to let transformers choose.
        random_init: If True, build the model from its config with random
            weights instead of downloading checkpoints. Use for plumbing smoke
            tests only -- routing will be near-uniform by construction, which
            is NOT evidence about real expert skew.
        max_memory: Optional per-device capacity cap handed to ``device_map``,
            e.g. ``{"0": "20GiB", "1": "90GiB", "cpu": "100GiB"}``. Keys are
            device indices as strings (or "cpu"). Use this when a GPU is shared
            with another job so accelerate does not claim memory that is already
            in use. Note that accelerate places weights only -- leave headroom
            for activations and KV cache on top of the model size.
    """

    name_or_path: str
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    trust_remote_code: bool = False
    attn_implementation: str | None = "sdpa"
    random_init: bool = False
    max_memory: dict[str, str] | None = None


@dataclass
class DataConfig:
    """Which corpus to run through the model.

    Attributes:
        dataset: Hugging Face dataset id. WikiText-2 is the docs.md default.
        subset: Dataset config name.
        split: Dataset split.
        seq_len: Tokens per sequence. Sequences are packed back-to-back from
            the concatenated corpus so no positions are wasted on padding.
        max_sequences: Cap on number of packed sequences (None = whole split).
        batch_size: Sequences per forward pass.
        seed: Only used if ``shuffle_documents`` is True.
        shuffle_documents: Shuffle documents before packing.
    """

    dataset: str = "wikitext"
    subset: str | None = "wikitext-2-raw-v1"
    split: str = "test"
    seq_len: int = 512
    max_sequences: int | None = 64
    batch_size: int = 4
    seed: int = 0
    shuffle_documents: bool = False


@dataclass
class ProfilerConfig:
    """How much detail to record.

    Attributes:
        record_trace: If True, write the full per-(token, layer, expert)
            dispatch trace to parquet. Stage 3's scheduler consumes this, so it
            defaults on. Row count is roughly
            ``n_tokens * n_moe_layers * top_k`` -- check the estimate the runner
            prints before profiling a long corpus.
        trace_flush_rows: Buffered rows before a parquet row-group is flushed.
        cross_check_router: If True, verify that indices recomputed from the
            router logits match the indices the router itself emitted. Costs a
            little time; catches silent routing-extraction bugs.
        max_new_tokens: 0 profiles a pure prefill/teacher-forced forward pass
            (what docs.md 4.1 describes). >0 additionally profiles autoregressive
            decode steps, which is the regime stage 3 schedules for.
    """

    record_trace: bool = True
    trace_flush_rows: int = 200_000
    cross_check_router: bool = True
    max_new_tokens: int = 0


@dataclass
class ClassifyConfig:
    """Hot/cold split parameters.

    Attributes:
        method: "top_fraction" marks the top ``value`` fraction of experts (by
            dispatch count) hot. "coverage" marks the smallest set of experts
            covering ``value`` fraction of all dispatches hot. "count" marks the
            top ``value`` experts hot.
        value: Interpreted per ``method``.
        per_layer: If True, the rule is applied independently within each MoE
            layer (matching how experts are actually resident per layer). If
            False, applied globally across all (layer, expert) pairs.
    """

    method: str = "top_fraction"
    value: float = 0.2
    per_layer: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"top_fraction", "coverage", "count"}:
            raise ValueError(f"unknown classify method: {self.method!r}")
        if self.method in {"top_fraction", "coverage"} and not 0.0 < self.value <= 1.0:
            raise ValueError(f"{self.method} requires 0 < value <= 1, got {self.value}")
        if self.method == "count" and self.value < 1:
            raise ValueError(f"count requires value >= 1, got {self.value}")


@dataclass
class RunConfig:
    """Top-level config: one profiling run."""

    run_name: str
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    output_dir: str = "data"

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(
            run_name=raw["run_name"],
            model=ModelConfig(**raw["model"]),
            data=DataConfig(**raw.get("data", {})),
            profiler=ProfilerConfig(**raw.get("profiler", {})),
            classify=ClassifyConfig(**raw.get("classify", {})),
            output_dir=raw.get("output_dir", "data"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / "runs" / self.run_name
