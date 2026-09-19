"""
SWORD v2 Ensemble — Package Init.

Provides the SWORDEnsembleDetector for multi-model weapon detection
using Weighted Boxes Fusion (WBF).
"""

from ensemble.ensemble_model import SWORDEnsembleDetector, CLASS_NAMES
from ensemble.wbf import weighted_boxes_fusion, nms

__all__ = [
    "SWORDEnsembleDetector",
    "CLASS_NAMES",
    "weighted_boxes_fusion",
    "nms",
]
