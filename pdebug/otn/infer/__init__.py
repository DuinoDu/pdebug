# Created by AI
"""Infer node registrations and helpers for OTN."""

# Ensure infer nodes are imported so they register with the manager.
from . import trajectory_analysis_node

__all__ = ["trajectory_analysis_node"]
