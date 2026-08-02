"""Perception-XAlpha Lite: research-only factor discovery."""

from .discovery import run_discovery
from .pit import align_point_in_time_fundamentals

__all__ = ["align_point_in_time_fundamentals", "run_discovery"]
__version__ = "0.1.0"

