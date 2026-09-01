"""External provider boundaries.

Provider modules are intentionally imported explicitly by their consumers.
Keeping this package initializer side-effect free avoids a provider/core import
cycle and makes lightweight modules safe to import in tests and tooling.
"""
