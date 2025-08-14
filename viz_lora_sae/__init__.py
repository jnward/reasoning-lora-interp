"""
LoRA → SAE → Category Flow Visualization Pipeline

This package provides tools for visualizing the information flow from rank-1 LoRA
adaptations through SAE features to semantic categories in mathematical reasoning models.
"""

__version__ = "0.1.0"

from .flows import (
    load_inputs_mode_a,
    load_inputs_mode_b,
    normalize_confidences,
    aggregate_middle_nodes,
)

from .figure import (
    build_sankey,
    build_category_bar,
    build_evidence_tiles,
)

__all__ = [
    "load_inputs_mode_a",
    "load_inputs_mode_b", 
    "normalize_confidences",
    "aggregate_middle_nodes",
    "build_sankey",
    "build_category_bar",
    "build_evidence_tiles",
]