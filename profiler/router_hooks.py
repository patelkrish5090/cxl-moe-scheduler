"""Discover and instrument the MoE routers of a Hugging Face model.

Why hooks on the *router* rather than on individual experts: in transformers
>= 5.x the experts of Mixtral / OLMoE / Qwen*-MoE are fused into stacked
parameters (e.g. ``MixtralExperts.gate_up_proj`` has shape
``[num_experts, 2 * intermediate, hidden]``) and are dispatched inside a single
module call, so there is no per-expert module to hook. Every one of these
architectures does, however, expose a router module whose forward returns a
``[n_tokens, num_experts]`` logits tensor and (for most) an integer
``[n_tokens, top_k]`` index tensor. That is the stable interception point.

This module produces only dispatch *counts* and index traces -- no timing and no
energy. Stage 2 supplies energy/latency numbers.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

try:  # torch is required at runtime but not for importing the dataclasses
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


_LAYER_IDX_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)")

# Class-name suffixes used by the MoE routers in transformers 5.x.
_ROUTER_CLASS_RE = re.compile(r"(TopKRouter|TopKGating|MoeGate|Router|Gating)$")

# Attribute names a SparseMoeBlock uses for its router child.
_ROUTER_ATTR_NAMES = {"gate", "router"}


def _layer_index_from_name(name: str) -> int:
    """Extract the transformer layer index from a module's qualified name.

    Returns -1 when the name carries no recognisable layer index.
    """
    match = _LAYER_IDX_RE.search(name)
    return int(match.group(1)) if match else -1


def _config_int(config: Any, *names: str) -> int | None:
    """First positive int found among ``names`` on ``config`` (or its text_config)."""
    for attr in names:
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return _config_int(text_config, *names)
    return None


@dataclass
class RouterSite:
    """One instrumented router (i.e. one MoE layer).

    Attributes:
        name: Fully qualified module name inside the model.
        layer_idx: Transformer layer index this router belongs to.
        num_experts: Number of experts this router chooses among.
        top_k: Experts selected per token.
        expert_weight_bytes: Bytes of expert parameters for a *single* expert in
            this layer, measured from the loaded tensors. 0 when the expert
            parameters could not be located (e.g. meta-device models). This is a
            size in bytes, never an energy figure.
        module: The live ``nn.Module`` (not serialised).
    """

    name: str
    layer_idx: int
    num_experts: int
    top_k: int
    expert_weight_bytes: int = 0
    module: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer_idx": self.layer_idx,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_weight_bytes": self.expert_weight_bytes,
        }


def _measure_expert_weight_bytes(moe_block: Any, num_experts: int) -> int:
    """Bytes of parameters belonging to one expert of ``moe_block``.

    Sums every parameter under the block that is *not* the router itself and
    divides by the expert count. Works for both fused expert parameters
    (``[num_experts, ...]`` stacks) and per-expert submodule lists. Returns 0 if
    nothing measurable is found or the parameters live on the meta device.
    """
    if nn is None or moe_block is None or num_experts <= 0:
        return 0
    total = 0
    for param_name, param in moe_block.named_parameters(recurse=True):
        # Skip router/gate weights: they are per-layer, not per-expert.
        head = param_name.split(".")[0]
        if head in _ROUTER_ATTR_NAMES:
            continue
        if param.device.type == "meta":
            return 0
        total += param.numel() * param.element_size()
    return total // num_experts


def discover_routers(model: Any) -> list[RouterSite]:
    """Find every MoE router in ``model``.

    A module is treated as a router when its class name matches the known
    router-class pattern, or when it is reachable as ``.gate``/``.router`` of a
    parent and carries a ``top_k`` attribute. Expert counts and top-k are read
    from the module when present and fall back to the model config.

    Raises:
        RuntimeError: if no routers are found -- profiling a model with no
            detected MoE layers would silently produce an empty log.
        ValueError: if a router reports ``top_k > num_experts``.
    """
    if nn is None:  # pragma: no cover
        raise RuntimeError("torch is required to discover routers")

    config = getattr(model, "config", None)
    cfg_experts = _config_int(config, "num_local_experts", "num_experts", "n_routed_experts")
    cfg_top_k = _config_int(config, "num_experts_per_tok", "top_k", "moe_top_k")

    modules = dict(model.named_modules())
    sites: list[RouterSite] = []
    seen: set[int] = set()

    for name, module in modules.items():
        cls_name = type(module).__name__
        attr_name = name.rsplit(".", 1)[-1]
        looks_like_router = (
            bool(_ROUTER_CLASS_RE.search(cls_name))
            or (attr_name in _ROUTER_ATTR_NAMES and hasattr(module, "top_k"))
            # transformers 4.x: MixtralSparseMoeBlock.gate is a bare nn.Linear
            # that returns only the [n_tokens, num_experts] logits tensor, with
            # no top_k attribute and no router class name.
            or (
                attr_name in _ROUTER_ATTR_NAMES
                and isinstance(module, nn.Linear)
                and cfg_experts is not None
                and module.out_features == cfg_experts
            )
        )
        if not looks_like_router or id(module) in seen:
            continue

        num_experts = getattr(module, "num_experts", None) or cfg_experts
        top_k = getattr(module, "top_k", None) or cfg_top_k
        if not isinstance(num_experts, int) or not isinstance(top_k, int):
            warnings.warn(
                f"skipping candidate router {name!r} ({cls_name}): could not determine "
                f"num_experts/top_k (got {num_experts!r}/{top_k!r})",
                stacklevel=2,
            )
            continue
        if top_k > num_experts:
            raise ValueError(f"router {name!r}: top_k={top_k} exceeds num_experts={num_experts}")

        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        moe_block = modules.get(parent_name)
        expert_bytes = _measure_expert_weight_bytes(moe_block, num_experts)

        seen.add(id(module))
        sites.append(
            RouterSite(
                name=name,
                layer_idx=_layer_index_from_name(name),
                num_experts=num_experts,
                top_k=top_k,
                expert_weight_bytes=expert_bytes,
                module=module,
            )
        )

    if not sites:
        raise RuntimeError(
            "no MoE routers found in this model. Either it is not a Mixture-of-Experts "
            "model, or its router class name is unrecognised -- inspect "
            "[type(m).__name__ for _, m in model.named_modules()] and extend "
            "_ROUTER_CLASS_RE in profiler/router_hooks.py."
        )

    sites.sort(key=lambda s: (s.layer_idx, s.name))
    return sites


def _iter_tensors(obj: Any) -> Iterable[Any]:
    """Yield tensors from an arbitrarily nested tuple/list/dict router output."""
    if torch is not None and isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item)


def extract_routing(output: Any, num_experts: int, top_k: int) -> tuple[Any, Any]:
    """Pull ``(indices, logits)`` out of a router's forward output.

    Handles the two shapes present in transformers 5.x:

    * ``(router_logits[T, E], router_scores[T, K], router_indices[T, K])``
      -- Mixtral / OLMoE / Qwen2-MoE / Qwen3-MoE.
    * ``(index_sorted_experts, batch_index, batch_gates, expert_size,
      logits[T, E])`` -- GraniteMoe, whose index tensors are 1-D and therefore
      correctly ignored here; indices are recomputed from the logits.

    Selection rules: ``indices`` is the first *integer* 2-D tensor whose second
    dim equals ``top_k``; ``logits`` is the first *floating* 2-D tensor whose
    second dim equals ``num_experts``. When no index tensor is present, indices
    are recomputed as ``topk(logits, top_k)``.

    Returns:
        ``(indices[T, K] int64, logits[T, E] or None)``.

    Raises:
        RuntimeError: when neither an index tensor nor a logits tensor is found.
    """
    indices = None
    logits = None
    for tensor in _iter_tensors(output):
        if tensor.ndim != 2:
            continue
        if indices is None and tensor.shape[1] == top_k and not tensor.is_floating_point():
            indices = tensor
        elif logits is None and tensor.shape[1] == num_experts and tensor.is_floating_point():
            logits = tensor

    if indices is None:
        if logits is None:
            shapes = [tuple(t.shape) for t in _iter_tensors(output)]
            raise RuntimeError(
                "router output contained neither an integer [T, top_k] index tensor nor a "
                f"float [T, {num_experts}] logits tensor; got tensor shapes {shapes}"
            )
        indices = torch.topk(logits, top_k, dim=-1).indices

    return indices.detach().to(torch.int64), (None if logits is None else logits.detach())


@dataclass
class _BatchContext:
    """Token bookkeeping for the forward pass currently in flight."""

    batch_size: int
    seq_len: int
    position_offset: int
    token_uid_base: int
    phase: str  # "prefill" or "decode"
    valid_mask_flat: Any = None  # optional bool [B*S] marking real (non-pad) tokens


class RouterProfiler:
    """Records per-token expert dispatch for every router in a model.

    Usage::

        with RouterProfiler(model) as prof:
            for batch in loader:
                prof.begin_batch(batch_size=B, seq_len=S, phase="prefill")
                model(**batch)
        counts = prof.counts  # int64 [n_sites, max_experts]

    The profiler is a plain forward-hook recorder: it never modifies the router
    output, so the model's own behaviour is unchanged.
    """

    def __init__(
        self,
        model: Any,
        sites: Sequence[RouterSite] | None = None,
        record_trace: bool = True,
        cross_check: bool = True,
    ) -> None:
        self.model = model
        self.sites: list[RouterSite] = list(sites) if sites is not None else discover_routers(model)
        self.record_trace = record_trace
        self.cross_check = cross_check

        self.max_experts = max(site.num_experts for site in self.sites)
        # counts[site_idx, expert_id] = number of token dispatches to that expert
        self.counts = np.zeros((len(self.sites), self.max_experts), dtype=np.int64)
        # tokens_per_site[site_idx] = number of tokens routed through that site
        self.tokens_per_site = np.zeros(len(self.sites), dtype=np.int64)

        self._handles: list[Any] = []
        self._ctx: _BatchContext | None = None
        self._next_token_uid = 0
        self._trace_chunks: list[dict[str, np.ndarray]] = []

        self.cross_check_total = 0
        self.cross_check_mismatch = 0
        self.shape_warnings = 0
        self.orphan_dispatches = 0

    # ---------------------------------------------------------------- lifecycle
    def __enter__(self) -> "RouterProfiler":
        self.attach()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.detach()

    def attach(self) -> None:
        """Register forward hooks on every discovered router (idempotent)."""
        if self._handles:
            return
        for site_idx, site in enumerate(self.sites):
            self._handles.append(site.module.register_forward_hook(self._make_hook(site_idx)))

    def detach(self) -> None:
        """Remove all hooks (idempotent)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ------------------------------------------------------------------ batches
    def begin_batch(
        self,
        batch_size: int,
        seq_len: int,
        phase: str = "prefill",
        position_offset: int = 0,
        valid_mask: Any = None,
    ) -> None:
        """Declare the token layout of the forward pass about to run.

        Routers see hidden states flattened to ``[batch_size * seq_len, hidden]``
        in row-major order, so row ``r`` corresponds to batch item ``r // seq_len``
        at sequence position ``position_offset + (r % seq_len)``.

        Args:
            batch_size: Sequences in the batch.
            seq_len: Query length of this forward pass (1 for a decode step).
            phase: "prefill" or "decode"; recorded in the trace.
            position_offset: Absolute position of the first query token, i.e. the
                KV-cache length for decode steps.
            valid_mask: Optional bool tensor ``[batch_size, seq_len]`` marking real
                (non-padding) tokens. Dispatches for masked-out tokens are dropped
                so padding cannot inflate the activation histogram.
        """
        flat_mask = None
        if valid_mask is not None:
            flat_mask = valid_mask.reshape(-1).to(torch.bool)
        self._ctx = _BatchContext(
            batch_size=batch_size,
            seq_len=seq_len,
            position_offset=position_offset,
            token_uid_base=self._next_token_uid,
            phase=phase,
            valid_mask_flat=flat_mask,
        )
        self._next_token_uid += batch_size * seq_len

    # -------------------------------------------------------------------- hooks
    def _make_hook(self, site_idx: int):
        site = self.sites[site_idx]

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            indices, logits = extract_routing(output, site.num_experts, site.top_k)
            self._record(site_idx, indices, logits)

        return hook

    def _record(self, site_idx: int, indices: Any, logits: Any) -> None:
        site = self.sites[site_idx]
        n_rows = int(indices.shape[0])
        ctx = self._ctx

        if self.cross_check and logits is not None:
            recomputed = torch.topk(logits.float(), site.top_k, dim=-1).indices
            same = (
                torch.sort(recomputed, dim=-1).values == torch.sort(indices, dim=-1).values
            ).all(dim=-1)
            self.cross_check_total += int(same.numel())
            self.cross_check_mismatch += int((~same).sum().item())

        keep = None
        if ctx is not None and ctx.valid_mask_flat is not None:
            mask = ctx.valid_mask_flat
            if mask.shape[0] == n_rows:
                keep = mask.to(indices.device)
            else:
                self.shape_warnings += 1

        counted = indices if keep is None else indices[keep]
        if counted.numel():
            bins = torch.bincount(counted.reshape(-1), minlength=site.num_experts)
            self.counts[site_idx, : site.num_experts] += bins.cpu().numpy().astype(np.int64)
        self.tokens_per_site[site_idx] += n_rows if keep is None else int(keep.sum().item())

        if not self.record_trace:
            return
        if ctx is None:
            self.orphan_dispatches += n_rows
            if self.orphan_dispatches == n_rows:  # warn once
                warnings.warn(
                    "router fired outside begin_batch(); trace rows dropped. Call "
                    "RouterProfiler.begin_batch() before each model forward pass.",
                    stacklevel=2,
                )
            return
        self._append_trace(site_idx, indices, ctx, keep, n_rows)

    def _append_trace(
        self, site_idx: int, indices: Any, ctx: _BatchContext, keep: Any, n_rows: int
    ) -> None:
        site = self.sites[site_idx]
        expected = ctx.batch_size * ctx.seq_len
        rows = np.arange(n_rows, dtype=np.int64)
        if n_rows != expected:
            # Do not guess a token layout we cannot justify: record positions as
            # -1 rather than silently mislabelling tokens.
            if self.shape_warnings < 5:
                warnings.warn(
                    f"router {site.name!r} saw {n_rows} rows but begin_batch() declared "
                    f"{ctx.batch_size}x{ctx.seq_len}={expected}; token positions for this "
                    "batch are recorded as -1.",
                    stacklevel=2,
                )
            self.shape_warnings += 1
            seq_pos = np.full(n_rows, -1, dtype=np.int32)
            batch_item = np.full(n_rows, -1, dtype=np.int16)
            token_uid = np.full(n_rows, -1, dtype=np.int64)
        else:
            batch_item = (rows // ctx.seq_len).astype(np.int16)
            seq_pos = (ctx.position_offset + rows % ctx.seq_len).astype(np.int32)
            token_uid = (ctx.token_uid_base + rows).astype(np.int64)

        idx_np = indices.cpu().numpy().astype(np.int16)
        if keep is not None:
            keep_np = keep.cpu().numpy().astype(bool)
            idx_np = idx_np[keep_np]
            batch_item = batch_item[keep_np]
            seq_pos = seq_pos[keep_np]
            token_uid = token_uid[keep_np]

        kept_rows, top_k = idx_np.shape
        if kept_rows == 0:
            return
        n = kept_rows * top_k
        self._trace_chunks.append(
            {
                "token_uid": np.repeat(token_uid, top_k),
                "batch_item": np.repeat(batch_item, top_k),
                "seq_pos": np.repeat(seq_pos, top_k),
                "layer_idx": np.full(n, site.layer_idx, dtype=np.int16),
                "site_idx": np.full(n, site_idx, dtype=np.int16),
                "expert_id": idx_np.reshape(-1),
                "slot_k": np.tile(np.arange(top_k, dtype=np.int8), kept_rows),
                "is_decode": np.full(n, ctx.phase == "decode", dtype=bool),
            }
        )

    # ------------------------------------------------------------------ outputs
    def pending_trace_rows(self) -> int:
        """Number of buffered trace rows not yet flushed to disk."""
        return sum(len(chunk["token_uid"]) for chunk in self._trace_chunks)

    def take_trace(self) -> dict[str, np.ndarray]:
        """Return and clear the buffered trace rows as a column dict."""
        if not self._trace_chunks:
            return {}
        keys = list(self._trace_chunks[0].keys())
        merged = {k: np.concatenate([c[k] for c in self._trace_chunks]) for k in keys}
        self._trace_chunks.clear()
        return merged

    def summary(self) -> dict[str, Any]:
        """Run-level sanity numbers, including routing cross-check results."""
        mismatch_rate = (
            self.cross_check_mismatch / self.cross_check_total if self.cross_check_total else None
        )
        return {
            "n_moe_layers": len(self.sites),
            "max_experts": int(self.max_experts),
            "total_dispatches": int(self.counts.sum()),
            "tokens_per_site": self.tokens_per_site.tolist(),
            "cross_check_rows": self.cross_check_total,
            "cross_check_mismatch": self.cross_check_mismatch,
            "cross_check_mismatch_rate": mismatch_rate,
            "shape_warnings": self.shape_warnings,
            "orphan_dispatches": self.orphan_dispatches,
        }
