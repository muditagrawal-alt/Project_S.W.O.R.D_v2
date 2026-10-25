"""Make the project root importable (ensemble/, tracking/, pose_pipeline.py) however pytest is invoked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
