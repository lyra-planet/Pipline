"""Public failure-diagnosis and closed-set repair policy boundary."""

try:
    from vetra_failure_repair import *  # noqa: F401,F403
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from ..vetra_failure_repair import *  # noqa: F401,F403
