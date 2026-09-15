"""On-disk formats for stage-1 outputs.

A profiling run writes one directory, ``data/runs/<run_name>/``:

===========================  ==============================================
``run_metadata.json``        config used, model/router topology, sanity stats
``expert_counts.csv``        [site_idx, layer_idx, expert_id, dispatch_count]
``hot_cold.csv``             classification table (one row per expert)
``layer_stats.csv``          per-layer skew statistics
``trace.parquet``            per-(token, layer, expert) dispatch trace
``plots/*.png``              histogram / heatmap / coverage curve
===========================  ==============================================

``trace.parquet`` is the hand-off to stage 3: it is the ordered stream of
expert requests a scheduler must serve. Column units are documented in
:data:`TRACE_SCHEMA_DOC`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRACE_SCHEMA_DOC: dict[str, str] = {
    "token_uid": "int64 - globally unique token index across the run, in issue order",
    "batch_item": "int16 - sequence index within its batch (-1 if unknown)",
    "seq_pos": "int32 - absolute position of the token in its sequence (-1 if unknown)",
    "layer_idx": "int16 - transformer layer index of the MoE block",
    "site_idx": "int16 - index into run_metadata.json['routers']",
    "expert_id": "int16 - expert selected within that layer",
    "slot_k": "int8 - which of the top-k slots this dispatch occupies (0 = highest scoring)",
    "is_decode": "bool - True if produced during autoregressive decode, False during prefill",
}


class TraceWriter:
    """Incremental parquet writer for the per-token dispatch trace.

    Rows are appended as row-groups so a long corpus never has to be held in
    memory. Falls back to CSV only if ``pyarrow`` is unavailable.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows_written = 0
        self._writer: Any = None
        self._schema: Any = None
        try:
            import pyarrow  # noqa: F401

            self._backend = "parquet"
        except Exception:
            self._backend = "csv"
            self.path = self.path.with_suffix(".csv")
            self._csv_header_written = False

    def write(self, columns: dict[str, np.ndarray]) -> int:
        """Append a chunk of trace rows. Returns rows written in this call."""
        if not columns:
            return 0
        frame = pd.DataFrame(columns)
        if self._backend == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self._writer is None:
                self._schema = table.schema
                self._writer = pq.ParquetWriter(self.path, self._schema, compression="zstd")
            else:
                table = table.cast(self._schema)
            self._writer.write_table(table)
        else:
            frame.to_csv(
                self.path,
                mode="a" if self._csv_header_written else "w",
                header=not self._csv_header_written,
                index=False,
            )
            self._csv_header_written = True
        self.rows_written += len(frame)
        return len(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def write_counts_csv(
    path: str | Path,
    counts: np.ndarray,
    layer_ids: list[int],
    experts_per_site: list[int],
) -> pd.DataFrame:
    """Write the raw ``[site, expert] -> dispatch_count`` table and return it."""
    rows = []
    for site_idx in range(counts.shape[0]):
        for expert_id in range(experts_per_site[site_idx]):
            rows.append(
                {
                    "site_idx": site_idx,
                    "layer_idx": int(layer_ids[site_idx]),
                    "expert_id": expert_id,
                    "dispatch_count": int(counts[site_idx, expert_id]),
                }
            )
    frame = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    """Write ``run_metadata.json``, converting numpy scalars to plain Python."""

    def default(obj: Any) -> Any:
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"not JSON serialisable: {type(obj)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=default), encoding="utf-8")


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load a completed run's outputs for inspection or downstream stages.

    Returns a dict with keys ``metadata``, ``counts``, ``hot_cold``,
    ``layer_stats``, and ``trace_path`` (which may be ``None``).
    """
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"no run_metadata.json in {run_dir}")

    def _csv(name: str) -> pd.DataFrame | None:
        target = run_dir / name
        return pd.read_csv(target) if target.exists() else None

    trace_path: Path | None = None
    for candidate in (run_dir / "trace.parquet", run_dir / "trace.csv"):
        if candidate.exists():
            trace_path = candidate
            break

    return {
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        "counts": _csv("expert_counts.csv"),
        "hot_cold": _csv("hot_cold.csv"),
        "layer_stats": _csv("layer_stats.csv"),
        "trace_path": trace_path,
    }


def counts_matrix_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """Rebuild the ``[n_sites, max_experts]`` count matrix from the CSV table."""
    n_sites = int(frame["site_idx"].max()) + 1
    n_experts = int(frame["expert_id"].max()) + 1
    matrix = np.zeros((n_sites, n_experts), dtype=np.int64)
    matrix[frame["site_idx"].to_numpy(), frame["expert_id"].to_numpy()] = frame[
        "dispatch_count"
    ].to_numpy()
    return matrix
