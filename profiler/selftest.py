"""Offline correctness checks for the stage-1 profiler. No downloads, CPU-only.

    python -m profiler.cli selftest

Builds a tiny randomly-initialised Mixtral from an in-code config (never touches
the Hub), then checks the profiler against an *independently* computed ground
truth: a separate pre-hook captures each router's input hidden states, and the
top-k selection is recomputed from scratch outside the profiler's code path. If
the two disagree, the hooking is wrong.

Note the deliberate limit of this test: an untrained router routes almost
uniformly, so this file proves the plumbing is correct and proves nothing about
real expert skew. Skew evidence has to come from a trained model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .classify import ClassifyConfig, classify, coverage_curve, gini, normalized_entropy
from .router_hooks import RouterProfiler, discover_routers, extract_routing

_PASS = 0
_FAIL = 0


def _check(name: str, condition: bool, detail: str = "", verbose: bool = True) -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        if verbose:
            print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def _tiny_mixtral():
    """A ~1M-parameter Mixtral built from an in-code config (no Hub access)."""
    import torch
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    config = MixtralConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        router_jitter_noise=0.0,
    )
    model = MixtralForCausalLM(config)
    model.eval()
    return model, config


def test_discovery(verbose: bool = True) -> None:
    print("\n[discovery]")
    model, config = _tiny_mixtral()
    sites = discover_routers(model)
    _check("one router per transformer layer", len(sites) == config.num_hidden_layers,
           f"found {len(sites)}, expected {config.num_hidden_layers}", verbose)
    _check("expert count read correctly",
           all(s.num_experts == config.num_local_experts for s in sites), verbose=verbose)
    _check("top_k read correctly",
           all(s.top_k == config.num_experts_per_tok for s in sites), verbose=verbose)
    _check("layer indices are 0..n-1",
           [s.layer_idx for s in sites] == list(range(config.num_hidden_layers)),
           f"got {[s.layer_idx for s in sites]}", verbose)

    # Expert weight size for Mixtral: gate_up_proj (2*I*H) + down_proj (I*H) per
    # expert, times element size. Checked against the config arithmetic.
    expected_bytes = 3 * config.intermediate_size * config.hidden_size * 4  # float32
    _check("expert weight bytes match config arithmetic",
           all(s.expert_weight_bytes == expected_bytes for s in sites),
           f"got {sites[0].expert_weight_bytes}, expected {expected_bytes}", verbose)


def test_extract_routing(verbose: bool = True) -> None:
    print("\n[extract_routing]")
    import torch

    n_tokens, n_experts, top_k = 7, 8, 2
    logits = torch.randn(n_tokens, n_experts)
    true_idx = torch.topk(logits, top_k, dim=-1).indices

    # Mixtral/OLMoE/Qwen shape: (logits, scores, indices)
    scores = torch.rand(n_tokens, top_k)
    idx, lg = extract_routing((logits, scores, true_idx), n_experts, top_k)
    _check("3-tuple form: indices taken verbatim", torch.equal(idx, true_idx.to(torch.int64)), verbose=verbose)
    _check("3-tuple form: logits recovered", lg is not None and torch.equal(lg, logits), verbose=verbose)
    _check("float scores[T,K] not mistaken for indices",
           not torch.equal(idx.float(), scores), verbose=verbose)

    # GraniteMoe shape: 1-D index tensors must be ignored, indices recomputed.
    granite_out = (
        torch.arange(n_tokens * top_k),      # index_sorted_experts, 1-D
        torch.arange(n_tokens * top_k),      # batch_index, 1-D
        torch.rand(n_tokens * top_k),        # batch_gates, 1-D
        [1] * n_experts,                     # expert_size, a list
        logits,                              # logits [T, E]
    )
    idx2, lg2 = extract_routing(granite_out, n_experts, top_k)
    _check("granite form: indices recomputed from logits",
           torch.equal(torch.sort(idx2, -1).values, torch.sort(true_idx, -1).values), verbose=verbose)
    _check("granite form: 1-D tensors ignored", lg2 is not None and torch.equal(lg2, logits), verbose=verbose)

    # transformers 4.x: the gate is a bare nn.Linear returning only logits.
    idx3, lg3 = extract_routing(logits, n_experts, top_k)
    _check("bare logits tensor (transformers 4.x gate) handled",
           torch.equal(torch.sort(idx3, -1).values, torch.sort(true_idx, -1).values)
           and lg3 is not None, verbose=verbose)

    try:
        extract_routing((torch.rand(3),), n_experts, top_k)
        _check("unrecognisable output raises", False, "no exception raised", verbose)
    except RuntimeError:
        _check("unrecognisable output raises", True, verbose=verbose)


def test_legacy_linear_gate_discovery(verbose: bool = True) -> None:
    """A transformers-4.x-style bare nn.Linear gate must still be discovered."""
    print("\n[legacy nn.Linear gate discovery]")
    import torch
    import torch.nn as nn

    class LegacyMoeBlock(nn.Module):
        """Mimics transformers 4.x MixtralSparseMoeBlock: gate is a plain Linear."""

        def __init__(self, hidden: int, n_experts: int) -> None:
            super().__init__()
            self.gate = nn.Linear(hidden, n_experts, bias=False)
            self.experts = nn.ModuleList(
                nn.Linear(hidden, hidden, bias=False) for _ in range(n_experts)
            )

        def forward(self, hidden_states):
            return self.gate(hidden_states.reshape(-1, hidden_states.shape[-1]))

    class LegacyLayer(nn.Module):
        def __init__(self, hidden: int, n_experts: int) -> None:
            super().__init__()
            self.block_sparse_moe = LegacyMoeBlock(hidden, n_experts)

        def forward(self, hidden_states):
            return self.block_sparse_moe(hidden_states)

    class LegacyModel(nn.Module):
        def __init__(self, hidden=16, n_experts=8, top_k=2, n_layers=2) -> None:
            super().__init__()
            self.layers = nn.ModuleList(LegacyLayer(hidden, n_experts) for _ in range(n_layers))
            self.config = type(
                "Cfg", (), {"num_local_experts": n_experts, "num_experts_per_tok": top_k}
            )()

        def forward(self, hidden_states):
            for layer in self.layers:
                layer(hidden_states)
            return hidden_states

    torch.manual_seed(5)
    model = LegacyModel()
    sites = discover_routers(model)
    _check("legacy gate discovered in every layer", len(sites) == 2,
           f"found {len(sites)}", verbose)
    _check("legacy gate: num_experts/top_k from config",
           all(s.num_experts == 8 and s.top_k == 2 for s in sites), verbose=verbose)
    _check("legacy gate: layer index parsed", [s.layer_idx for s in sites] == [0, 1],
           f"got {[s.layer_idx for s in sites]}", verbose)
    # Expert bytes must exclude the gate: 8 experts x 16x16 float32 params each.
    _check("legacy gate excluded from expert weight bytes",
           all(s.expert_weight_bytes == 16 * 16 * 4 for s in sites),
           f"got {sites[0].expert_weight_bytes}, expected {16 * 16 * 4}", verbose)

    n_tokens = 12
    with torch.no_grad(), RouterProfiler(model, sites=sites, record_trace=True) as prof:
        prof.begin_batch(batch_size=1, seq_len=n_tokens, phase="prefill")
        model(torch.randn(1, n_tokens, 16))
    _check("legacy gate: counts recorded",
           int(prof.counts.sum()) == n_tokens * 2 * 2,
           f"got {int(prof.counts.sum())}, expected {n_tokens * 2 * 2}", verbose)


def test_counts_against_ground_truth(verbose: bool = True) -> None:
    """The core test: profiler counts must equal an independent recomputation."""
    print("\n[counts vs independent ground truth]")
    import torch
    import torch.nn.functional as F

    model, config = _tiny_mixtral()
    sites = discover_routers(model)

    # Independent path: capture each router's INPUT and redo top-k by hand.
    truth = np.zeros((len(sites), config.num_local_experts), dtype=np.int64)
    truth_pairs: list[tuple[int, int]] = []  # (site_idx, expert_id) in dispatch order

    def make_pre_hook(site_idx: int):
        def pre_hook(module, args):
            hidden = args[0].reshape(-1, config.hidden_size)
            logits = F.linear(hidden, module.weight)
            picks = torch.topk(logits.float().softmax(-1), config.num_experts_per_tok, -1).indices
            for expert_id in picks.reshape(-1).tolist():
                truth[site_idx, expert_id] += 1
            for row in picks.tolist():
                for expert_id in row:
                    truth_pairs.append((site_idx, expert_id))
        return pre_hook

    handles = [s.module.register_forward_pre_hook(make_pre_hook(i)) for i, s in enumerate(sites)]

    torch.manual_seed(1)
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad(), RouterProfiler(model, sites=sites, record_trace=True) as prof:
        prof.begin_batch(batch_size=batch_size, seq_len=seq_len, phase="prefill")
        model(input_ids=input_ids)

    for handle in handles:
        handle.remove()

    _check("counts match independent recomputation exactly",
           np.array_equal(prof.counts, truth),
           f"profiler={prof.counts.tolist()}\n        truth   ={truth.tolist()}", verbose)

    expected_dispatches = batch_size * seq_len * config.num_hidden_layers * config.num_experts_per_tok
    _check("total dispatches = tokens x layers x top_k",
           int(prof.counts.sum()) == expected_dispatches,
           f"got {int(prof.counts.sum())}, expected {expected_dispatches}", verbose)
    _check("cross-check found zero mismatches",
           prof.cross_check_mismatch == 0 and prof.cross_check_total > 0,
           f"{prof.cross_check_mismatch}/{prof.cross_check_total}", verbose)
    _check("no shape warnings", prof.shape_warnings == 0, verbose=verbose)

    trace = prof.take_trace()
    _check("trace row count = total dispatches",
           len(trace["token_uid"]) == expected_dispatches,
           f"got {len(trace['token_uid'])}", verbose)

    # Trace must reproduce the count matrix.
    from_trace = np.zeros_like(prof.counts)
    np.add.at(from_trace, (trace["site_idx"].astype(int), trace["expert_id"].astype(int)), 1)
    _check("trace aggregates back to the count matrix",
           np.array_equal(from_trace, prof.counts), verbose=verbose)

    _check("token_uid covers exactly one id per token",
           sorted(set(trace["token_uid"].tolist())) == list(range(batch_size * seq_len)),
           verbose=verbose)
    _check("seq_pos within [0, seq_len)",
           trace["seq_pos"].min() == 0 and trace["seq_pos"].max() == seq_len - 1, verbose=verbose)
    _check("batch_item within [0, batch_size)",
           trace["batch_item"].min() == 0 and trace["batch_item"].max() == batch_size - 1,
           verbose=verbose)
    _check("all rows flagged prefill", not trace["is_decode"].any(), verbose=verbose)


def test_decode_phase(verbose: bool = True) -> None:
    print("\n[decode phase]")
    import torch

    model, config = _tiny_mixtral()
    sites = discover_routers(model)
    batch_size, seq_len, n_new = 2, 8, 3

    torch.manual_seed(2)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    with torch.no_grad(), RouterProfiler(model, sites=sites, record_trace=True) as prof:
        prof.begin_batch(batch_size=batch_size, seq_len=seq_len, phase="prefill")
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_ids = out.logits[:, -1:].argmax(-1)
        for step in range(n_new):
            prof.begin_batch(batch_size=batch_size, seq_len=1, phase="decode",
                             position_offset=seq_len + step)
            out = model(input_ids=next_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_ids = out.logits[:, -1:].argmax(-1)

    trace = prof.take_trace()
    n_layers, top_k = config.num_hidden_layers, config.num_experts_per_tok
    expected_prefill = batch_size * seq_len * n_layers * top_k
    expected_decode = batch_size * n_new * n_layers * top_k
    _check("prefill rows counted correctly",
           int((~trace["is_decode"]).sum()) == expected_prefill,
           f"got {int((~trace['is_decode']).sum())}, expected {expected_prefill}", verbose)
    _check("decode rows counted correctly",
           int(trace["is_decode"].sum()) == expected_decode,
           f"got {int(trace['is_decode'].sum())}, expected {expected_decode}", verbose)
    decode_pos = trace["seq_pos"][trace["is_decode"]]
    _check("decode positions continue past the prompt",
           decode_pos.min() == seq_len and decode_pos.max() == seq_len + n_new - 1,
           f"range [{decode_pos.min()}, {decode_pos.max()}]", verbose)
    _check("no shape warnings during decode", prof.shape_warnings == 0, verbose=verbose)


def test_padding_mask(verbose: bool = True) -> None:
    print("\n[padding mask]")
    import torch

    model, config = _tiny_mixtral()
    sites = discover_routers(model)
    batch_size, seq_len, n_valid = 2, 10, 6

    torch.manual_seed(3)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    valid = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    valid[:, :n_valid] = True

    with torch.no_grad(), RouterProfiler(model, sites=sites, record_trace=True) as prof:
        prof.begin_batch(batch_size=batch_size, seq_len=seq_len, phase="prefill", valid_mask=valid)
        model(input_ids=input_ids)

    expected = batch_size * n_valid * config.num_hidden_layers * config.num_experts_per_tok
    _check("masked tokens excluded from counts",
           int(prof.counts.sum()) == expected,
           f"got {int(prof.counts.sum())}, expected {expected}", verbose)
    trace = prof.take_trace()
    _check("masked tokens excluded from trace", len(trace["token_uid"]) == expected, verbose=verbose)
    _check("no trace row beyond the valid prefix",
           int(trace["seq_pos"].max()) == n_valid - 1, verbose=verbose)


def test_classify(verbose: bool = True) -> None:
    print("\n[classification]")
    # Layer 0: experts 0-1 dominate. Layer 1: perfectly uniform.
    counts = np.array([[100, 80, 5, 5, 4, 3, 2, 1], [10, 10, 10, 10, 10, 10, 10, 10]], dtype=np.int64)

    res = classify(counts, ClassifyConfig(method="top_fraction", value=0.25, per_layer=True))
    hot0 = set(res.table.query("site_idx == 0 and is_hot")["expert_id"])
    _check("top_fraction picks the busiest experts", hot0 == {0, 1}, f"got {hot0}", verbose)
    _check("hot count is ceil(fraction * n)", int(res.table["is_hot"].sum()) == 4, verbose=verbose)

    res_cov = classify(counts, ClassifyConfig(method="coverage", value=0.9, per_layer=True))
    stats0 = res_cov.per_layer_stats.iloc[0]
    n_hot0 = int(stats0["n_hot"])
    _check("coverage rule reaches its target",
           stats0["hot_dispatch_share"] >= 0.9, f"got {stats0['hot_dispatch_share']:.3f}", verbose)
    # Minimality: the hot set must be the smallest that clears the target, i.e.
    # one fewer expert must fall short. Asserted against the curve rather than a
    # hardcoded size, so the check stays honest if the fixture changes.
    curve0 = coverage_curve(counts[0])
    _check("coverage rule is minimal (one fewer expert falls short)",
           n_hot0 >= 1 and (n_hot0 == 1 or curve0[n_hot0 - 2] < 0.9),
           f"n_hot={n_hot0}, curve={curve0.round(4).tolist()}", verbose)

    res_cnt = classify(counts, ClassifyConfig(method="count", value=3, per_layer=True))
    _check("count rule marks exactly n per layer",
           (res_cnt.per_layer_stats["n_hot"] == 3).all(), verbose=verbose)

    _check("gini: skewed layer > uniform layer",
           res.per_layer_stats.iloc[0]["gini"] > res.per_layer_stats.iloc[1]["gini"], verbose=verbose)
    _check("gini of uniform vector is ~0", abs(gini(counts[1])) < 1e-9, verbose=verbose)
    _check("normalised entropy of uniform vector is 1.0",
           abs(normalized_entropy(counts[1]) - 1.0) < 1e-9, verbose=verbose)
    _check("normalised entropy of one-hot vector is 0.0",
           abs(normalized_entropy(np.array([7, 0, 0, 0]))) < 1e-9, verbose=verbose)

    curve = coverage_curve(counts[0])
    _check("coverage curve is monotonic and ends at 1.0",
           bool(np.all(np.diff(curve) >= -1e-12)) and abs(curve[-1] - 1.0) < 1e-9, verbose=verbose)

    zero = classify(np.zeros((1, 4), dtype=np.int64), ClassifyConfig(method="top_fraction", value=0.5))
    _check("unused experts are never hot", int(zero.table["is_hot"].sum()) == 0, verbose=verbose)

    global_res = classify(counts, ClassifyConfig(method="count", value=2, per_layer=False))
    _check("global split marks n experts across all layers, not per layer",
           int(global_res.table["is_hot"].sum()) == 2, verbose=verbose)

    weighted = classify(counts, ClassifyConfig(method="count", value=2, per_layer=True),
                        expert_weight_bytes=[1000, 2000])
    _check("hot/cold weight bytes accounted per layer",
           weighted.overall["hot_weight_bytes"] == 2 * 1000 + 2 * 2000
           and weighted.overall["cold_weight_bytes"] == 6 * 1000 + 6 * 2000,
           f"hot={weighted.overall['hot_weight_bytes']}, cold={weighted.overall['cold_weight_bytes']}",
           verbose)


def test_io_roundtrip(verbose: bool = True) -> None:
    print("\n[log I/O]")
    from .activation_log import (TraceWriter, counts_matrix_from_frame, load_run,
                                 write_counts_csv, write_metadata)

    counts = np.array([[100, 80, 5, 5], [10, 10, 10, 10]], dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "runs" / "t"
        frame = write_counts_csv(run_dir / "expert_counts.csv", counts, [0, 1], [4, 4])
        _check("counts CSV has one row per expert", len(frame) == 8, verbose=verbose)
        _check("count matrix round-trips through CSV",
               np.array_equal(counts_matrix_from_frame(frame), counts), verbose=verbose)

        with TraceWriter(run_dir / "trace.parquet") as writer:
            chunk = {
                "token_uid": np.arange(6, dtype=np.int64),
                "batch_item": np.zeros(6, dtype=np.int16),
                "seq_pos": np.arange(6, dtype=np.int32),
                "layer_idx": np.zeros(6, dtype=np.int16),
                "site_idx": np.zeros(6, dtype=np.int16),
                "expert_id": np.arange(6, dtype=np.int16) % 4,
                "slot_k": np.zeros(6, dtype=np.int8),
                "is_decode": np.zeros(6, dtype=bool),
            }
            writer.write(chunk)
            writer.write(chunk)  # second row-group exercises the append path
        _check("trace writer appended both chunks", writer.rows_written == 12, verbose=verbose)

        import pandas as pd
        read_back = pd.read_parquet(writer.path) if writer.path.suffix == ".parquet" else pd.read_csv(writer.path)
        _check("trace round-trips off disk", len(read_back) == 12, f"got {len(read_back)}", verbose)

        res = classify(counts, ClassifyConfig(method="count", value=1))
        res.table.to_csv(run_dir / "hot_cold.csv", index=False)
        res.per_layer_stats.to_csv(run_dir / "layer_stats.csv", index=False)
        write_metadata(run_dir / "run_metadata.json", {
            "run_name": "t",
            "classification": res.overall,
            "model_topology": {"expert_weight_bytes_per_layer": [0, 0]},
        })
        loaded = load_run(run_dir)
        _check("load_run finds every artefact",
               loaded["counts"] is not None and loaded["hot_cold"] is not None
               and loaded["layer_stats"] is not None and loaded["trace_path"] is not None,
               verbose=verbose)
        _check("metadata survives numpy types",
               loaded["metadata"]["classification"]["n_hot_total"] == 2, verbose=verbose)


def test_plots(verbose: bool = True) -> None:
    print("\n[plots]")
    counts = np.array([[100, 80, 5, 5, 4, 3, 2, 1], [10, 12, 9, 11, 10, 10, 8, 10]], dtype=np.int64)
    res = classify(counts, ClassifyConfig(method="top_fraction", value=0.25),
                   layer_ids=[0, 1], experts_per_site=[8, 8], expert_weight_bytes=[10**6, 10**6])
    with tempfile.TemporaryDirectory() as tmp:
        from . import plots as plots_mod
        written = plots_mod.plot_all(res, counts, [8, 8], Path(tmp) / "plots", " - selftest")
        _check("three plots written", len(written) == 3, verbose=verbose)
        _check("plot files are non-empty",
               all(p.exists() and p.stat().st_size > 1000 for p in written), verbose=verbose)
    table = plots_mod.format_summary_table(res)
    _check("summary table renders", "OVERALL" in table and "gini" in table, verbose=verbose)
    if verbose:
        print("\n" + table)


def main(verbose: bool = True) -> int:
    global _PASS, _FAIL
    _PASS = _FAIL = 0
    print("stage-1 profiler selftest (offline, CPU, no model downloads)")
    test_discovery(verbose)
    test_extract_routing(verbose)
    test_legacy_linear_gate_discovery(verbose)
    test_counts_against_ground_truth(verbose)
    test_decode_phase(verbose)
    test_padding_mask(verbose)
    test_classify(verbose)
    test_io_roundtrip(verbose)
    test_plots(verbose)
    print(f"\n{'=' * 60}\n{_PASS} passed, {_FAIL} failed\n{'=' * 60}")
    if _FAIL == 0:
        print("Plumbing verified. NOTE: this uses an untrained model, so it says\n"
              "nothing about real expert skew -- that needs a trained checkpoint.")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
