"""Stage 1: MoE expert activation profiler.

Instruments a Hugging Face MoE model's routers to record which experts each
token is dispatched to, then classifies experts as hot or cold.

Units convention for this package (see CLAUDE.md):
  * counts are dimensionless integers (number of token-to-expert dispatches)
  * sizes are bytes, named ``*_bytes``
No energy or latency values are produced at this stage.
"""

__all__ = ["config", "router_hooks", "data", "runner", "activation_log", "classify", "plots"]
