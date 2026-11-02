"""Detection modules for S.W.O.R.D. v2: Weapon WBF Ensemble and Person Detector."""
from .weapon_ensemble import WeaponEnsembleDetector, SWORD_WEAPON_CLASSES
from .person_detector import PersonDetector
from .wbf import weighted_boxes_fusion, nms

__all__ = [
    "WeaponEnsembleDetector",
    "PersonDetector",
    "SWORD_WEAPON_CLASSES",
    "weighted_boxes_fusion",
    "nms",
]

