"""Association modules for S.W.O.R.D. v2: Person <-> Weapon Spatio-Temporal Matcher."""
from .person_weapon_matcher import SpatioTemporalAssociator, AssociationRecord, compute_iow_and_distance

__all__ = ["SpatioTemporalAssociator", "AssociationRecord", "compute_iow_and_distance"]

