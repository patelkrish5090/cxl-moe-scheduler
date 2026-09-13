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
python -m profiler.cli analyze data/runs/<name> --phase decode --max-sites 0
```

`reclassify` re-splits hot/cold at a new threshold from the saved counts, so
sweeping thresholds never requires re-running the model.

`analyze` measures temporal locality — see [Locality and the layer
split](#locality-and-the-layer-split).

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

## Locality and the layer split

`python -m profiler.cli analyze data/runs/<name>` answers the question the
histogram cannot: not *which* experts are used most, but whether the experts a
token needs are the ones the previous tokens needed. It compares three policies
at each cache size — static pinning by frequency, LRU, and Belady MIN — and the
`LRU-static` column is what runtime adaptation is worth, i.e. what stage 3's
scheduler has to beat.

Layers are not interchangeable in that measurement. A layer whose **working set**
— the fewest experts covering 99 % of its dispatches — is no larger than `top_k`
is fully resident in the smallest cache worth simulating, so every policy scores
~100 % on it at every capacity. Those layers are **cache-trivial**. They are real
routing behaviour, not an error, but averaging them into the policy comparison
pulls every policy toward 100 % and shrinks the measured `LRU-static` gap without
carrying any information about placement.

`analyze` therefore prints the layer classification (section `[2]`) and then the
locality comparison for both groups. Use `--layers`:

| value | tables printed |
| --- | --- |
| `both` (default) | all layers, then diverse layers only |
| `all` | every layer, averaged together |
| `diverse` | only layers with a working set larger than `top_k` |
| `trivial` | only the cache-trivial layers |

**Correction (superseded claim):** an earlier version of this section claimed
Mixtral-8x7B-v0.1 layers 1–15 "route every token to experts 0 and 1" in both
prefill and decode, and cited `scripts/inspect_routers.py models/mixtral
--load` as confirmation. That check only inspects the router's weight
tensors (it never runs a forward pass), so it could not actually have
confirmed a *routing* behaviour. The real, clean runs (after fixing the
`device_map="auto"` NaN-corruption bug documented in this project's stage-1
history — sharding Mixtral across 2 GPUs silently poisoned router logits with
NaN, and `torch.topk` does not raise on NaN, it deterministically biases
toward low indices, which looks exactly like "always picks experts 0, 1")
show only mild skew: Gini 0.069–0.115 across the prefill/decode Mixtral runs,
not the near-total collapse the original claim described. Mixtral is NOT
cache-trivial in the way this section previously implied.

Mild-but-real skew (clearly nonzero Gini, not the near-total collapse above)
is itself consistent with Mixtral's own published routing analysis
(Jiang et al. 2024, "Mixtral of Experts," arXiv:2401.04088, sec. 5 "Routing
analysis," Figure 7): the paper reports per-expert selection proportions
close to the 1/8 uniform-sampling reference line across domains, a direct
consequence of the auxiliary load-balancing loss used during training. A
flat-zero Gini would still mean the hooking is wrong (docs.md 6 checkpoint
1's actual warning sign), but a modest, nonzero Gini for a model trained this
way is the expected result, not a symptom of a broken profiler -- see
docs.md 6's checkpoint 1 note for the project-level implication.

Two flags interact with this:

- `--phase decode` restricts to autoregressive decode, the regime a serving
  scheduler operates in. It also recomputes the dispatch counts from the filtered
  trace, so skew, coverage and locality all describe the same tokens rather than
  mixing phase-local traffic with run-wide frequencies.
- `--max-sites 0` simulates every layer instead of an evenly-spaced sample of 4.
  Each layer group is subsampled independently, so with a small `--max-sites` the
  two tables are averages over different layer samples — compare them as
  populations, not row by row.

## Dense vs MoE

The original problem statement's objectives ask to "analyze Transformer and
MoE memory access patterns" -- everything above profiles the MoE half. To
give the MoE skew a real point of comparison rather than an assertion, set
`model.architecture: "dense"` (default: `"moe"`) to profile a genuinely
dense (non-MoE) Transformer instead. `configs/gpt2_dense_decode.json` /
`gpt2_dense_wikitext2.json` do this against a real GPT-2 checkpoint.

```bash
python -m profiler.cli run configs/gpt2_dense_wikitext2.json
python -m profiler.cli run configs/gpt2_dense_decode.json
```

**How it works**: a dense Transformer has no gating function to hook --
`router_hooks.discover_dense_sites` finds each decoder layer's plain FFN/MLP
block instead (matched by the attribute name `mlp`, e.g. GPT-2/Llama/Mistral),
and `router_hooks.DenseProfiler` records every real token that reaches that
layer as a dispatch to a degenerate single "expert 0" (`num_experts=1`,
`top_k=1`). This is not a routing decision -- there is no gate -- but it pushes
a real dense-model forward pass through the EXACT same trace/hot_cold.csv
schema stage 1 already uses for MoE models, so `classify.py`'s gini/entropy,
`plots.py`'s heatmap, and the dashboard's activation-skew summary all work on
a dense run unmodified, and the two can be compared on identical axes.

**Expected, and confirmed real** result: Gini = 0.000, every layer 100% "hot"
(there is no cold tier, because there is nothing to split -- every layer's
single FFN is used by every token, always). This is not a placeholder or a
degenerate edge case being tolerated -- it IS the finding: unlike an MoE
model's real, measurable per-layer skew (see "Locality and the layer split"
above), a dense Transformer has no hot/cold split to exploit at all, so
expert tiering has nothing to offer it. That is the direct, data-backed
answer to why this project's tiering strategy is MoE-specific.

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

`python -m profiler.cli selftest` (66 checks) builds tiny models in-process —
no Hub access — and verifies profiler counts against an **independently**
computed ground truth: a separate pre-hook captures each router's input hidden
states and redoes the top-k from scratch, outside the profiler's code path. It
also covers both router return shapes, decode-phase position bookkeeping,
padding-mask exclusion, every threshold method, log I/O round-trips, and (a
real, tiny GPT2LMHeadModel, not a mock) `discover_dense_sites` +
`DenseProfiler`'s dense-mode path — confirms every real token dispatches to
the degenerate single expert at every layer, and that classification comes
out trivially all-hot with Gini exactly 0, the expected result for a model
with nothing to route.

`tests/test_runner_integration.py` runs `runner.run()` end to end with only the
two network calls stubbed, then checks that `trace.parquet` aggregates back to
`expert_counts.csv` exactly.

Additionally, every run cross-checks at profiling time that indices recomputed
from the router logits match the indices the router emitted, and reports the
mismatch rate. A non-zero rate means the model uses something other than plain
top-k (grouped or bias-corrected routing, e.g. DeepSeek-V3 style) and
`extract_routing()` needs review for that architecture before its numbers are
trusted.
