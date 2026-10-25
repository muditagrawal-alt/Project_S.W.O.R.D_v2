"""
Single source of truth for SWORD v2 class IDs and names.

Class IDs are what the YOLO label files and the trained checkpoints use; they never change.
Only the human-readable names were inconsistent across dataset.yaml, folder names, scripts and
checkpoints ("Sniper Dataset", "assault rifle", "Sniper Rifle", "Sniper", ...). Import names from
here instead of redefining them.
"""

# Display names (reports, figures, UI).
CLASS_NAMES = [
    "Pistol", "Shotgun", "Assault Rifle", "Grenade", "Knife",
    "Sword", "Sniper Rifle", "SMG", "Rocket Launcher", "Machine Gun",
]

# Machine-friendly names (configs, column names, file names).
CANONICAL_NAMES = [
    "pistol", "shotgun", "assault_rifle", "grenade", "knife",
    "sword", "sniper_rifle", "smg", "rocket_launcher", "machine_gun",
]

# Names stored inside the trained checkpoints and dataset/dataset.yaml (kept for reference only).
LEGACY_NAMES = [
    "Pistol", "Shotgun", "assault rifle", "grenade", "knife",
    "sword", "Sniper Dataset", "SMG", "rocket launcher", "Machine gun",
]

NUM_CLASSES = len(CLASS_NAMES)
