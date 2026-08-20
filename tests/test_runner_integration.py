"""End-to-end check of ``profiler.runner.run`` without touching the network.

Only the two network-dependent calls are stubbed -- model download and corpus
download. Everything else (router discovery, hooking, batching, decode steps,
trace flushing, classification, CSV/JSON/parquet writing, plotting) runs for
real, so this exercises the same code path an HPC run takes.

    python tests/test_runner_integration.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiler import data as data_mod  # noqa: E402
from profiler import runner as runner_mod  # noqa: E402
from profiler.activation_log import load_run  # noqa: E402
from profiler.config import RunConfig  # noqa: E402

VOCAB_SIZE = 256
N_LAYERS = 3
N_EXPERTS = 8
TOP_K = 2


def _fake_model_loader(cfg: RunConfig):
    """Build a tiny Mixtral in-process instead of downloading one."""
    import torch
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    hf_config = MixtralConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        max_position_embeddings=256,
        router_jitter_noise=0.0,
    )
    model = MixtralForCausalLM(hf_config)
    model.eval()
    return model, None


def _fake_token_stream(cfg, tokenizer) -> np.ndarray:
    return data_mod.synthetic_token_stream(VOCAB_SIZE, n_tokens=2048, seed=7)


def main() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            failures.append(name)
            print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))

    real_loader = runner_mod.load_model_and_tokenizer
    real_stream = data_mod.load_token_stream
    runner_mod.load_model_and_tokenizer = _fake_model_loader
    data_mod.load_token_stream = _fake_token_stream

    tmp_dir = tempfile.mkdtemp()
    try:
        seq_len, max_seqs, batch_size, n_new = 64, 8, 4, 2
        cfg = RunConfig.from_dict(
            {
                "run_name": "integration",
                "model": {"name_or_path": "in-memory-tiny-mixtral", "dtype": "float32",
                          "device_map": "cpu"},
                "data": {"seq_len": seq_len, "max_sequences": max_seqs, "batch_size": batch_size},
                "profiler": {"record_trace": True, "trace_flush_rows": 500,
                             "cross_check_router": True, "max_new_tokens": n_new},
                "classify": {"method": "top_fraction", "value": 0.25, "per_layer": True},
                "output_dir": tmp_dir,
            }
        )

        print("\n[running profiler.runner.run]")
        out = runner_mod.run(cfg, verbose=True)
        run_dir = Path(out["run_dir"])

        print("\n[artefacts]")
        for name in ("run_metadata.json", "expert_counts.csv", "hot_cold.csv", "layer_stats.csv",
                     "trace.parquet"):
            check(f"{name} written", (run_dir / name).exists())
        for name in ("activation_histogram.png", "activation_heatmap.png", "coverage_curve.png"):
            path = run_dir / "plots" / name
            check(f"plots/{name} non-empty", path.exists() and path.stat().st_size > 1000)

        print("\n[numbers]")
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        prefill = max_seqs * seq_len
        decode = max_seqs * n_new
        expected_dispatches = (prefill + decode) * N_LAYERS * TOP_K

        check("prefill token count correct",
              metadata["workload"]["prefill_tokens"] == prefill,
              f"got {metadata['workload']['prefill_tokens']}, expected {prefill}")
        check("decode token count correct",
              metadata["workload"]["decode_tokens"] == decode,
              f"got {metadata['workload']['decode_tokens']}, expected {decode}")
        check("total dispatches = tokens x layers x top_k",
              metadata["profiler_summary"]["total_dispatches"] == expected_dispatches,
              f"got {metadata['profiler_summary']['total_dispatches']}, "
              f"expected {expected_dispatches}")
        check("trace rows = total dispatches",
              metadata["trace"]["rows"] == expected_dispatches,
              f"got {metadata['trace']['rows']}")
        check("multi-flush path exercised (rows > flush threshold)",
              expected_dispatches > cfg.profiler.trace_flush_rows,
              f"{expected_dispatches} vs {cfg.profiler.trace_flush_rows}")
        check("router cross-check clean",
              metadata["profiler_summary"]["cross_check_mismatch"] == 0)
        check("no shape warnings", metadata["profiler_summary"]["shape_warnings"] == 0)
        check("no orphan dispatches", metadata["profiler_summary"]["orphan_dispatches"] == 0)
        check("every MoE layer discovered", metadata["model_topology"]["n_moe_layers"] == N_LAYERS)
        check("expert weight bytes measured",
              all(b > 0 for b in metadata["model_topology"]["expert_weight_bytes_per_layer"]))

        print("\n[trace <-> counts consistency]")
        import pandas as pd

        trace = pd.read_parquet(run_dir / "trace.parquet")
        counts_csv = pd.read_csv(run_dir / "expert_counts.csv")
        from_trace = trace.groupby(["site_idx", "expert_id"]).size().rename("n").reset_index()
        merged = counts_csv.merge(from_trace, on=["site_idx", "expert_id"], how="left").fillna({"n": 0})
        check("per-expert trace counts equal expert_counts.csv",
              bool((merged["dispatch_count"] == merged["n"]).all()),
              merged[merged["dispatch_count"] != merged["n"]].to_string())
        check("trace schema columns as documented",
              set(trace.columns) == set(metadata["trace"]["schema"].keys()),
              f"got {sorted(trace.columns)}")
        check("decode rows present and flagged",
              int(trace["is_decode"].sum()) == decode * N_LAYERS * TOP_K,
              f"got {int(trace['is_decode'].sum())}")
        check("token_uid unique per token across the run",
              trace["token_uid"].nunique() == prefill + decode,
              f"got {trace['token_uid'].nunique()}, expected {prefill + decode}")

        print("\n[load_run]")
        loaded = load_run(run_dir)
        check("load_run returns all tables",
              all(loaded[k] is not None for k in ("counts", "hot_cold", "layer_stats", "trace_path")))
        check("hot_cold covers every expert",
              len(loaded["hot_cold"]) == N_LAYERS * N_EXPERTS,
              f"got {len(loaded['hot_cold'])}")

        print("\n[untrained-model honesty note]")
        check("random/untrained caveat recorded in metadata notes "
              "(skew here is not real evidence)",
              any("dispatches" in n for n in metadata["notes"]))
        entropy = metadata["classification"]["normalized_entropy_overall"]
        print(f"        normalised entropy = {entropy:.3f} "
              f"(near 1.0 expected: this model is untrained)")

    finally:
        runner_mod.load_model_and_tokenizer = real_loader
        data_mod.load_token_stream = real_stream
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("integration test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
