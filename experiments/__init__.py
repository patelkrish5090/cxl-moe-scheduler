"""Stage 4 harness: runs a stage-1 trace through three configs -- HBM-only,
HBM+CXL naive, HBM+CXL energy-aware -- and logs throughput, latency, and
total energy for each (docs.md 4.6). This is orchestration, not new
simulation logic: every config is a specific (cache_capacity, policy) call
into the already-validated stage-3 simulator (scheduler/simulate.py).

See experiments/README.md for the config -> capacity mapping and unit
conventions.
"""
