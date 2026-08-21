"""End-to-end stage-1 profiling run.

Loads a Hugging Face MoE model, streams a corpus through it under
:class:`~profiler.router_hooks.RouterProfiler`, classifies experts hot/cold, and
writes the run directory described in :mod:`profiler.activation_log`.

Nothing here produces energy or latency figures; stage 2 owns those.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import activation_log, data, plots
from .activation_log import TraceWriter, write_counts_csv, write_metadata
from .classify import classify
from .config import RunConfig
from .router_hooks import RouterProfiler, discover_routers


def _resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"unknown dtype {name!r}")
    return dtype


def load_model_and_tokenizer(cfg: RunConfig) -> tuple[Any, Any]:
    """Load the model in eval mode plus its tokenizer.

    With ``model.random_init`` the weights are initialised randomly from the
    repo's config instead of downloaded. That path is for plumbing tests only:
    an untrained router routes near-uniformly, so its activation histogram says
    nothing about real expert skew.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg.model
    tokenizer_source = model_cfg.tokenizer_name_or_path or model_cfg.name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=model_cfg.trust_remote_code
        )
    except (ValueError, OSError) as exc:
        if "backend tokenizer" in str(exc) or "Can't load tokenizer" in str(exc):
            raise RuntimeError(
                f"could not build a tokenizer from {tokenizer_source!r}.\n"
                "Two different causes produce this same message:\n"
                "  1. The repo ships no tokenizer files at all (weights-only test "
                "repos such as hf-internal-testing/Mixtral-tiny are like this). Set "
                "model.tokenizer_name_or_path in the run config to a repo that has "
                "one, with a vocabulary no larger than the model's vocab_size.\n"
                "  2. The repo ships a SentencePiece tokenizer.model with no "
                "pre-built tokenizer.json, so transformers must convert it:\n"
                "         pip install sentencepiece protobuf\n"
                "     (Mixtral-8x7B needs this; OLMoE does not.)"
            ) from exc
        raise
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"trust_remote_code": model_cfg.trust_remote_code}
    dtype = _resolve_dtype(model_cfg.dtype)
    if dtype != "auto":
        kwargs["dtype"] = dtype
    if model_cfg.attn_implementation:
        kwargs["attn_implementation"] = model_cfg.attn_implementation

    if model_cfg.random_init:
        hf_config = AutoConfig.from_pretrained(
            model_cfg.name_or_path, trust_remote_code=model_cfg.trust_remote_code
        )
        model = AutoModelForCausalLM.from_config(hf_config, **kwargs)
        if model_cfg.device_map not in (None, "auto"):
            model = model.to(model_cfg.device_map)
    else:
        if model_cfg.device_map:
            kwargs["device_map"] = model_cfg.device_map
        if model_cfg.max_memory:
            # accelerate wants int keys for GPU indices, "cpu"/"disk" as strings.
            kwargs["max_memory"] = {
                (int(k) if k.isdigit() else k): v for k, v in model_cfg.max_memory.items()
            }
        model = AutoModelForCausalLM.from_pretrained(model_cfg.name_or_path, **kwargs)

    model.eval()
    return model, tokenizer


def _gpu_memory_report() -> list[dict[str, Any]]:
    """Free/total memory per visible CUDA device, in bytes."""
    import torch

    if not torch.cuda.is_available():
        return []
    report = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        report.append(
            {
                "index": i,
                "name": torch.cuda.get_device_properties(i).name,
                "free_bytes": int(free),
                "total_bytes": int(total),
            }
        )
    return report


def _device_placement(model: Any) -> dict[str, int]:
    """Parameter-count per device, so a sharded placement is visible in metadata."""
    placement: dict[str, int] = {}
    for param in model.parameters():
        key = str(param.device)
        placement[key] = placement.get(key, 0) + param.numel()
    return placement


def _first_param_device(model: Any) -> Any:
    for param in model.parameters():
        return param.device
    import torch

    return torch.device("cpu")


def run(cfg: RunConfig, verbose: bool = True) -> dict[str, Any]:
    """Execute one profiling run and write its output directory.

    Args:
        cfg: The run configuration.
        verbose: Print progress and the per-layer summary table.

    Returns:
        Dict with ``run_dir``, ``result`` (:class:`ClassificationResult`),
        ``counts``, and ``metadata``.
    """
    import torch

    started_at = time.time()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log(f"[1/6] loading model {cfg.model.name_or_path} "
        f"(dtype={cfg.model.dtype}, device_map={cfg.model.device_map}, "
        f"random_init={cfg.model.random_init})")
    model, tokenizer = load_model_and_tokenizer(cfg)
    device = _first_param_device(model)
    log(f"      loaded on {device}")

    log("[2/6] discovering MoE routers")
    sites = discover_routers(model)
    layer_ids = [s.layer_idx for s in sites]
    experts_per_site = [s.num_experts for s in sites]
    expert_bytes = [s.expert_weight_bytes for s in sites]
    total_expert_bytes = sum(b * e for b, e in zip(expert_bytes, experts_per_site))
    log(
        f"      {len(sites)} MoE layers | experts/layer={experts_per_site[0]} "
        f"| top_k={sites[0].top_k} | expert weight size="
        f"{expert_bytes[0] / 1e6:.1f} MB each, {total_expert_bytes / 1e9:.2f} GB total"
    )

    log(f"[3/6] loading corpus {cfg.data.dataset}/{cfg.data.subset} split={cfg.data.split}")
    token_stream = data.load_token_stream(cfg.data, tokenizer)
    sequences = data.pack_sequences(token_stream, cfg.data.seq_len, cfg.data.max_sequences)

    # A token id at or above the embedding size indexes out of bounds. On CPU
    # that is an IndexError; on CUDA it is an async device-side assert that
    # surfaces later with an unrelated stack trace. Check it here instead.
    model_vocab = getattr(model.config, "vocab_size", None)
    max_token_id = int(sequences.max())
    if isinstance(model_vocab, int) and max_token_id >= model_vocab:
        raise RuntimeError(
            f"tokenizer produced id {max_token_id} but the model's vocab_size is "
            f"{model_vocab}. The tokenizer ({cfg.model.tokenizer_name_or_path or cfg.model.name_or_path!r}) "
            f"does not match the model ({cfg.model.name_or_path!r}). Set "
            "model.tokenizer_name_or_path to a compatible tokenizer."
        )
    n_prefill_tokens = int(sequences.size)
    n_decode_tokens = sequences.shape[0] * cfg.profiler.max_new_tokens
    est_rows = (n_prefill_tokens + n_decode_tokens) * len(sites) * sites[0].top_k
    log(
        f"      {len(token_stream)} tokens -> {sequences.shape[0]} sequences x "
        f"{cfg.data.seq_len} tokens; estimated trace rows: {est_rows:,}"
    )

    trace_path = run_dir / "trace.parquet"
    writer = TraceWriter(trace_path) if cfg.profiler.record_trace else None

    log("[4/6] profiling")
    profiler = RouterProfiler(
        model,
        sites=sites,
        record_trace=cfg.profiler.record_trace,
        cross_check=cfg.profiler.cross_check_router,
    )
    n_batches = int(np.ceil(sequences.shape[0] / cfg.data.batch_size))
    forward_seconds = 0.0

    with torch.no_grad(), profiler:
        for batch_idx, batch in enumerate(data.iter_batches(sequences, cfg.data.batch_size)):
            input_ids = torch.from_numpy(batch).to(device)
            batch_size, seq_len = input_ids.shape

            profiler.begin_batch(batch_size=batch_size, seq_len=seq_len, phase="prefill")
            t0 = time.time()
            outputs = model(input_ids=input_ids, use_cache=cfg.profiler.max_new_tokens > 0)
            forward_seconds += time.time() - t0

            if cfg.profiler.max_new_tokens > 0:
                past = outputs.past_key_values
                next_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                for step in range(cfg.profiler.max_new_tokens):
                    profiler.begin_batch(
                        batch_size=batch_size,
                        seq_len=1,
                        phase="decode",
                        position_offset=seq_len + step,
                    )
                    t0 = time.time()
                    outputs = model(
                        input_ids=next_ids, past_key_values=past, use_cache=True
                    )
                    forward_seconds += time.time() - t0
                    past = outputs.past_key_values
                    next_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if writer is not None and profiler.pending_trace_rows() >= cfg.profiler.trace_flush_rows:
                writer.write(profiler.take_trace())

            if verbose and (batch_idx % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1):
                done = int(profiler.counts.sum())
                log(f"      batch {batch_idx + 1}/{n_batches}  dispatches={done:,}")

    if writer is not None:
        writer.write(profiler.take_trace())
        writer.close()
        log(f"      trace: {writer.rows_written:,} rows -> {writer.path}")

    summary = profiler.summary()
    if summary["total_dispatches"] == 0:
        raise RuntimeError(
            "no expert dispatches were recorded. The routers were discovered but never "
            "fired -- check that the model actually contains MoE layers on the executed path."
        )
    if summary["cross_check_mismatch"]:
        rate = summary["cross_check_mismatch_rate"]
        log(
            f"      WARNING: router cross-check mismatch on {rate:.2%} of token rows. "
            "The model's routing is not plain top-k over raw logits (e.g. grouped or "
            "bias-corrected routing). Counts still come from the router's own emitted "
            "indices where available; review extract_routing() before trusting them."
        )

    log("[5/6] classifying hot/cold")
    result = classify(
        profiler.counts,
        cfg.classify,
        layer_ids=layer_ids,
        experts_per_site=experts_per_site,
        expert_weight_bytes=expert_bytes,
    )

    write_counts_csv(run_dir / "expert_counts.csv", profiler.counts, layer_ids, experts_per_site)
    result.table.to_csv(run_dir / "hot_cold.csv", index=False)
    result.per_layer_stats.to_csv(run_dir / "layer_stats.csv", index=False)

    tokens_profiled = n_prefill_tokens + n_decode_tokens
    metadata: dict[str, Any] = {
        "run_name": cfg.run_name,
        "created_unix": started_at,
        "config": cfg.to_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            # Free/total bytes per visible GPU at the end of the run. Recorded
            # for reproducibility on a shared box; not an energy measurement.
            "gpu_memory_bytes": _gpu_memory_report(),
        },
        "device_placement": _device_placement(model),
        "model_topology": {
            "n_moe_layers": len(sites),
            "top_k": sites[0].top_k,
            "experts_per_layer": experts_per_site,
            "expert_weight_bytes_per_layer": expert_bytes,
            "total_expert_weight_bytes": total_expert_bytes,
        },
        "routers": [s.to_dict() for s in sites],
        "workload": {
            "corpus_tokens": int(len(token_stream)),
            "sequences": int(sequences.shape[0]),
            "seq_len": cfg.data.seq_len,
            "prefill_tokens": n_prefill_tokens,
            "decode_tokens": int(n_decode_tokens),
            "tokens_profiled": int(tokens_profiled),
            # Wall-clock of the instrumented forward passes only. This is a
            # profiling-run timing on this machine, NOT a throughput benchmark
            # and NOT an input to any energy calculation.
            "forward_wall_seconds": round(forward_seconds, 3),
        },
        "profiler_summary": summary,
        "classification": result.overall,
        "trace": {
            "path": str(writer.path.name) if writer else None,
            "rows": writer.rows_written if writer else 0,
            "schema": activation_log.TRACE_SCHEMA_DOC,
        },
        "notes": [
            "Counts are token-to-expert dispatches; a token dispatched to top_k experts "
            "contributes top_k counts.",
            "No energy or latency figures are produced at stage 1.",
        ],
    }
    if cfg.model.random_init:
        metadata["notes"].append(
            "random_init=True: weights are UNTRAINED. Routing skew from this run is an "
            "artefact of random initialisation and is not evidence about real MoE "
            "activation behaviour."
        )
    write_metadata(run_dir / "run_metadata.json", metadata)

    log("[6/6] plotting")
    suffix = f" - {cfg.run_name}"
    written = plots.plot_all(
        result, profiler.counts, experts_per_site, run_dir / "plots", title_suffix=suffix
    )
    for path in written:
        log(f"      {path}")

    if verbose:
        print()
        print(plots.format_summary_table(result))
        print()
        print(f"run directory: {run_dir.resolve()}")

    return {"run_dir": run_dir, "result": result, "counts": profiler.counts, "metadata": metadata}
