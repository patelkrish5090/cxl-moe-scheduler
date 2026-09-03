"""Why do some Mixtral layers route every token to the same two experts?

    python scripts/diagnose_collapse.py data/runs/mixtral_8x7b_decode

Reads only trace.parquet. Loads no model and needs no GPU.

The mixtral_8x7b_decode run showed layers 1-11 with 6 of 8 experts unused and
both survivors at exactly 50.0% of dispatches -- every token taking the same
pair. That is either a real property of Mixtral's router or an artifact of the
generated text being degenerate (a base model decoding greedily for 256 tokens
often falls into a repetition loop, and a repeated token routes identically).

The two are separable without re-running anything, because the trace holds both
phases. Prefill consumes real WikiText tokens and cannot be repetitive; decode
consumes the model's own output and can be. So:

    collapsed in prefill AND decode -> real router behaviour
    collapsed in decode only        -> degenerate generation, decode half suspect

Section [3] tests generation degeneracy directly: it builds a per-position
routing signature across all layers at once and counts how many distinct
signatures a sequence produces. Repetitive text yields far fewer distinct
signatures than it has positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative count vector. 0 = uniform, ->1 = skewed."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    sorted_counts = np.sort(counts.astype(np.float64))
    n = sorted_counts.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * sorted_counts).sum()) / (n * total) - (n + 1.0) / n)


def phase_stats(frame: pd.DataFrame, n_experts: int) -> dict[str, float]:
    """Dispatch concentration for one layer in one phase."""
    if frame.empty:
        return {"dispatches": 0, "used": 0, "gini": float("nan"), "top2": float("nan")}
    counts = np.bincount(frame["expert_id"].to_numpy(), minlength=n_experts)
    total = counts.sum()
    top2 = np.sort(counts)[::-1][:2].sum() / total
    return {
        "dispatches": int(total),
        "used": int((counts > 0).sum()),
        "gini": gini(counts),
        "top2": float(top2),
    }


def routing_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (sequence, position): how many distinct whole-model routings occur?

    A signature is the tuple of experts chosen at every layer for one token. If
    generation has fallen into a repetition loop, consecutive positions repeat
    their signature and the distinct count collapses well below the position
    count. Real text produces a near-unique signature per position.
    """
    if frame.empty:
        return pd.DataFrame()
    # token_uid is globally unique in issue order; batch_item is NOT a sequence
    # id when batch_size is 1, because every sequence is then batch item 0.
    ordered = frame.sort_values(["token_uid", "layer_idx", "slot_k"])
    key = ordered.groupby("token_uid")["expert_id"].apply(
        lambda s: hash(tuple(s.to_numpy().tolist()))
    ).reset_index(name="signature")

    # Recover sequence boundaries from seq_pos restarting, since consecutive
    # token_uids run continuously across sequence ends.
    positions = ordered.groupby("token_uid")["seq_pos"].first().reset_index(name="seq_pos")
    key = key.merge(positions, on="token_uid").sort_values("token_uid")
    key["seq"] = (key["seq_pos"].diff().fillna(1) <= 0).cumsum()

    rows = []
    for item, group in key.groupby("seq"):
        n_positions = len(group)
        n_distinct = group["signature"].nunique()
        # Consecutive-repeat rate: the clearest fingerprint of a decode loop.
        repeats = (group["signature"].to_numpy()[1:] == group["signature"].to_numpy()[:-1]).mean() \
            if n_positions > 1 else float("nan")
        rows.append({
            "seq": int(item),
            "positions": n_positions,
            "distinct_routings": n_distinct,
            "distinct_frac": n_distinct / n_positions if n_positions else float("nan"),
            "consecutive_repeat": repeats,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--experts", type=int, default=8,
                        help="experts per layer (default 8, Mixtral)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    trace = run_dir / "trace.parquet"
    if not trace.exists():
        trace = run_dir / "trace.csv"
    if not trace.exists():
        print(f"no trace found in {run_dir}")
        return 1

    frame = pd.read_parquet(trace) if trace.suffix == ".parquet" else pd.read_csv(trace)
    print(f"trace: {trace}  ({len(frame):,} rows)")

    if "is_decode" not in frame.columns:
        print("trace has no is_decode column -- cannot split phases.")
        return 1

    prefill = frame[~frame["is_decode"]]
    decode = frame[frame["is_decode"]]
    print(f"prefill rows: {len(prefill):,}   decode rows: {len(decode):,}")

    print("\n[1] PER-LAYER CONCENTRATION, SPLIT BY PHASE")
    print("    'used' = experts receiving at least one dispatch (of "
          f"{args.experts});  'top2' = share taken by the two busiest.")
    print()
    print("    layer |  prefill: used  gini   top2 |   decode: used  gini   top2 | verdict")
    print("    " + "-" * 84)

    verdicts: dict[str, list[int]] = {"both": [], "decode_only": [], "prefill_only": [], "healthy": []}
    for layer in sorted(frame["layer_idx"].unique()):
        p = phase_stats(prefill[prefill["layer_idx"] == layer], args.experts)
        d = phase_stats(decode[decode["layer_idx"] == layer], args.experts)
        # "Collapsed" = the two busiest experts absorb essentially everything.
        p_collapsed = p["top2"] > 0.99
        d_collapsed = d["top2"] > 0.99
        if p_collapsed and d_collapsed:
            verdict, bucket = "COLLAPSED both phases -> real router", "both"
        elif d_collapsed:
            verdict, bucket = "collapsed in DECODE only -> suspect", "decode_only"
        elif p_collapsed:
            verdict, bucket = "collapsed in prefill only", "prefill_only"
        else:
            verdict, bucket = "healthy", "healthy"
        verdicts[bucket].append(int(layer))
        print(f"    {layer:5d} | {p['used']:12d} {p['gini']:6.3f} {p['top2']:6.1%} |"
              f" {d['used']:11d} {d['gini']:6.3f} {d['top2']:6.1%} | {verdict}")

    print("\n[2] SUMMARY")
    for bucket, label in (
        ("both", "collapsed in BOTH phases (real router behaviour)"),
        ("decode_only", "collapsed in DECODE only (generation artifact)"),
        ("prefill_only", "collapsed in PREFILL only (unexpected)"),
        ("healthy", "healthy in both"),
    ):
        layers = verdicts[bucket]
        if layers:
            print(f"    {len(layers):2d} layers {label}")
            print(f"       {layers}")

    collapsed = sorted(set(verdicts["both"]) | set(verdicts["decode_only"]))
    if collapsed:
        print("\n[2b] WHICH EXPERTS DO THE COLLAPSED LAYERS PICK?")
        print("    Tied router logits make torch.topk fall back to index order, so a")
        print("    dead router always yields the LOWEST expert ids -- {0, 1} for")
        print("    top_k=2. Layer-specific pairs instead mean the weights are real.")
        print()
        pairs: dict[tuple[int, ...], list[int]] = {}
        for layer in collapsed:
            sub = frame[frame["layer_idx"] == layer]
            counts = np.bincount(sub["expert_id"].to_numpy(), minlength=args.experts)
            chosen = tuple(int(e) for e in np.argsort(counts)[::-1][:2] if counts[e] > 0)
            chosen = tuple(sorted(chosen))
            pairs.setdefault(chosen, []).append(layer)
            print(f"    layer {layer:3d} -> experts {chosen}")
        print()
        lowest = tuple(range(2))
        if set(pairs) == {lowest}:
            print(f"    ALL collapsed layers pick {lowest}, the lowest ids.")
            print("    -> CONSISTENT WITH tied router logits (dead gates), but NOT proof")
            print("       of it: a checkpoint can also genuinely favour the first two")
            print("       experts. This test cannot tell those apart. Settle it by")
            print("       reading the weights:")
            print("         python scripts/inspect_routers.py <model_dir> --load")
            print("       On Mixtral-8x7B-v0.1 that came back clean -- routers load with")
            print("       zero missing keys and non-degenerate weights -- so there the")
            print("       collapse in layers 1-15 is real routing, not a loading fault.")
        elif len(pairs) == 1:
            only = next(iter(pairs))
            print(f"    All collapsed layers pick the same pair {only}, but it is not")
            print(f"    {lowest}. Unusual: consistent with a shared upstream cause")
            print("    rather than index tie-breaking.")
        else:
            print(f"    {len(pairs)} distinct pairs across {len(collapsed)} layers:")
            for pair, layers in sorted(pairs.items()):
                print(f"      {pair}: layers {layers}")
            print("    -> Layer-specific pairs mean the gates hold real, distinct")
            print("       weights. The collapse is genuine routing behaviour, however")
            print("       extreme it looks.")

    print("\n[3] IS THE GENERATED TEXT DEGENERATE?")
    print("    Distinct whole-model routing signatures per generated position.")
    print("    A repetition loop drives distinct_frac toward 0 and")
    print("    consecutive_repeat toward 1.")
    print()
    sig_decode = routing_signatures(decode)
    if sig_decode.empty:
        print("    no decode rows")
    else:
        print(sig_decode.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        mean_frac = sig_decode["distinct_frac"].mean()
        mean_rep = sig_decode["consecutive_repeat"].mean()
        print(f"\n    mean distinct_frac      = {mean_frac:.3f}")
        print(f"    mean consecutive_repeat = {mean_rep:.3f}")
        print()
        if mean_rep > 0.5:
            print("    -> DEGENERATE. Most generated tokens route identically to their")
            print("       predecessor, i.e. the model is emitting a repetition loop.")
            print("       The decode locality numbers are measuring that loop, not")
            print("       serving behaviour. Re-run with sampling enabled.")
        elif mean_frac < 0.5:
            print("    -> PARTIALLY degenerate: many positions share a routing.")
            print("       Treat the decode locality numbers as an upper bound.")
        else:
            print("    -> Generation looks varied. Decode routing reflects real")
            print("       token-to-token variation, so the locality numbers stand.")

    # The same signature test on prefill is the control: real corpus text should
    # always look varied, so a low number here would indict the trace itself.
    sig_prefill = routing_signatures(prefill)
    if not sig_prefill.empty:
        print(f"\n    CONTROL (prefill, real WikiText tokens):")
        print(f"      mean distinct_frac      = {sig_prefill['distinct_frac'].mean():.3f}")
        print(f"      mean consecutive_repeat = {sig_prefill['consecutive_repeat'].mean():.3f}")
        print("      Prefill is real corpus text, so these are what 'varied' looks like.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
