# Stage 1 — Expert activation profiler

Implements docs.md 4.1 (profiler) and 4.2 (hot/cold classifier). Produces the
activation evidence and the per-token expert-request trace that stages 2–4
consume. **No energy or latency numbers are produced here** — stage 2 owns those.

## What it does

Hooks every MoE router in a Hugging Face model and records, for each token,
which experts it was dispatched to. Then splits experts into hot and cold under
a configurable threshold and writes the histogram / heatmap / coverage plots that
validation checkpoint 1 in docs.md calls for.

## Why hooks on the router, not on the experts

In `transformers` ≥ 5.x the experts of Mixtral / OLMoE / Qwen-MoE are **fused**
into stacked parameters — `MixtralExperts.gate_up_proj` has shape
`[num_experts, 2*intermediate, hidden]` and all experts are dispatched inside a
single module call. There is no per-expert module to hook. Every one of these
architectures does expose a router module (`MixtralTopKRouter`,
`OlmoeTopKRouter`, `Qwen3MoeTopKRouter`, `GraniteMoeTopKGating`, …) whose forward
returns a `[n_tokens, num_experts]` logits tensor and, for most, an integer
`[n_tokens, top_k]` index tensor. That is the stable interception point, and
`profiler/router_hooks.py` handles both return shapes.

The hooks are pure recorders — they never modify the router output, so model
behaviour is unchanged.

## Commands

```bash
python -m profiler.cli selftest                       # offline checks, no downloads
python tests/test_runner_integration.py               # end-to-end, no downloads
python -m profiler.cli run configs/olmoe_1b7b.json    # a real profiling run
python -m profiler.cli inspect data/runs/<name>       # re-print a run's summary
python -m profiler.cli reclassify data/runs/<name> --method coverage --value 0.9
```

`reclassify` re-splits hot/cold at a new threshold from the saved counts, so
sweeping thresholds never requires re-running the model.

## Running on the HPC box

Setup is a **separate conda env**, not the existing `teto` env:

```bash
bash scripts/setup_env.sh && conda activate astera
```

`teto` is missing `transformers`, `datasets`, `accelerate`, `safetensors` and
`pyarrow`, and carries numpy 2.5.2 / pandas 3.0.5. Installing into it risks pip
resolving numpy/pandas downwards underneath whatever is currently holding ~88 GB
on GPU 0.

### Hugging Face access

`allenai/OLMoE-1B-7B-0924`, `hf-internal-testing/Mixtral-tiny` and WikiText-2 are
public and need no token. The "unauthenticated requests" warning is harmless for
those, though a token raises the rate limit -- which matters on a shared NAT,
where the anonymous per-IP limit is shared with everyone else on the network.

`mistralai/Mixtral-8x7B-v0.1` is **gated**: accept the terms on the model page
while logged in, and export a token, or the download fails with 401/403.

```bash
hf auth login                      # or: export HF_TOKEN=hf_xxxx
```

Request access early -- approval is usually instant but can queue.

### GPU selection

`scripts/run_stage1.sh` pins to **physical GPU 1** (`CUDA_VISIBLE_DEVICES=1`),
because GPU 0 was 88 GB / 98 GB occupied at probe time. Note that
`CUDA_VISIBLE_DEVICES` **renumbers** devices: physical GPU 1 becomes `cuda:0`
inside the process, which is what `configs/olmoe_1b7b.json` and the `max_memory`
keys refer to. Once GPU 0 is free:

```bash
ASTERA_GPUS=0,1 bash scripts/run_stage1.sh mixtral
```

For a shared GPU, set `model.max_memory` in the run config so accelerate does not
claim memory another job is using — e.g. `{"0": "88GiB", "cpu": "150GiB"}`.
accelerate places *weights* only, so leave headroom for activations and KV cache
on top of the model size.

Each run records per-GPU free/total memory and the parameter-count-per-device
placement into `run_metadata.json` under `environment` and `device_placement`.

## Version compatibility

Router discovery handles both generations:

- **transformers 5.x** — `*TopKRouter` / `*TopKGating` classes returning a tuple.
- **transformers 4.x** — `SparseMoeBlock.gate` is a bare `nn.Linear` returning
  only `[n_tokens, num_experts]` logits, with no `top_k` attribute and no router
  class name. Detected by matching `out_features` against the config's expert
  count; top-k is then recomputed from the logits.

Both forms are covered by the selftest.

## Run directory layout

`data/runs/<run_name>/`

| File | Contents |
| --- | --- |
| `run_metadata.json` | config, router topology, workload, sanity stats, trace schema |
| `expert_counts.csv` | `site_idx, layer_idx, expert_id, dispatch_count` |
| `hot_cold.csv` | one row per expert: count, share, rank, `is_hot`, `expert_weight_bytes` |
| `layer_stats.csv` | per layer: gini, normalised entropy, hot share, unused experts |
| `trace.parquet` | per-`(token, layer, expert)` dispatch trace — **the stage-3 input** |
| `plots/*.png` | histogram, heatmap, coverage curve |

### `trace.parquet` schema

| Column | Type | Meaning |
| --- | --- | --- |
| `token_uid` | int64 | globally unique token index, in issue order |
| `batch_item` | int16 | sequence index within its batch (−1 if unknown) |
| `seq_pos` | int32 | absolute position in the sequence (−1 if unknown) |
| `layer_idx` | int16 | transformer layer index of the MoE block |
| `site_idx` | int16 | index into `run_metadata.json["routers"]` |
| `expert_id` | int16 | expert selected within that layer |
| `slot_k` | int8 | which top-k slot (0 = highest scoring) |
| `is_decode` | bool | True during autoregressive decode, False during prefill |

Sorted by `token_uid` this is exactly the ordered stream of expert requests the
stage-3 scheduler must serve.

## Units

Counts are dimensionless integers: one token dispatched to `top_k` experts
contributes `top_k` counts. Sizes are bytes and always carry a `_bytes` suffix.
`workload.forward_wall_seconds` in the metadata is the wall clock of the
instrumented forward passes on the profiling machine — it is **not** a
throughput benchmark and is **not** an input to any energy calculation.

## Reading the output

Three numbers decide whether stage 1 worked (docs.md §6, checkpoint 1):

- **`gini_overall`** — 0 = every expert used equally, 1 = one expert takes all.
- **`normalized_entropy_overall`** — 1.0 = uniform, 0.0 = one expert.
- **`coverage_curve.png`** — how many experts per layer must stay resident to
  serve 80/90/95 % of dispatches. Where the mean crosses 90 % is the defensible
  hot-set size, and it sizes stage 3's HBM budget directly.

A **flat** distribution (gini ≈ 0, entropy ≈ 1) on a **trained** model means the
hooking is wrong, not that the model is unusual.

On an **untrained** model (`random_init: true`, or `configs/smoke_tiny.json`) a
flat distribution is the *correct* result — an untrained router routes
near-uniformly. The integration test measures normalised entropy ≈ 0.99 on a
random-weight Mixtral, which is the expected control. Runs made this way get a
caveat written into `run_metadata.json["notes"]`; never cite their skew as
evidence.

## Threshold methods

`classify.method` in the run config:

- `top_fraction` — top `value` fraction of experts by count are hot (default 0.2).
- `coverage` — smallest expert set covering `value` of all dispatches.
- `count` — top `value` experts are hot.

`per_layer: true` applies the rule within each layer independently, which matches
how experts are actually resident per layer. Experts with zero dispatches are
never marked hot, and ties break toward the lower expert id so the split is
deterministic.

## Correctness checks

`python -m profiler.cli selftest` (56 checks) builds tiny models in-process —
no Hub access — and verifies profiler counts against an **independently**
computed ground truth: a separate pre-hook captures each router's input hidden
states and redoes the top-k from scratch, outside the profiler's code path. It
also covers both router return shapes, decode-phase position bookkeeping,
padding-mask exclusion, every threshold method, and log I/O round-trips.

`tests/test_runner_integration.py` runs `runner.run()` end to end with only the
two network calls stubbed, then checks that `trace.parquet` aggregates back to
`expert_counts.csv` exactly.

Additionally, every run cross-checks at profiling time that indices recomputed
from the router logits match the indices the router emitted, and reports the
mismatch rate. A non-zero rate means the model uses something other than plain
top-k (grouped or bias-corrected routing, e.g. DeepSeek-V3 style) and
`extract_routing()` needs review for that architecture before its numbers are
trusted.
