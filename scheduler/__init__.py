"""Stage 3: energy-aware expert-fetch scheduling.

Consumes stage 1's activation trace (real per-token expert dispatches) and
stage 2's tier model (real gem5/DRAMSim3 latency, bandwidth, and energy
figures) to simulate two scheduling policies over the same trace:

  naive        -- fetch every cold expert immediately, no energy weighting.
                  This is the comparison point: what capacity/latency-only
                  systems effectively do (docs.md 4.5).
  energy-aware -- same LRU cache, but gates *when* a cold fetch happens
                  against a replenishing power budget, deferring it under
                  budget pressure rather than always fetching immediately.

See scheduler/README.md for the model and its explicit simplifications.
"""
