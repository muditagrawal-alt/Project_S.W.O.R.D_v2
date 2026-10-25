"""
"Hard" pseudo-labelling pass (low confidence + test-time augmentation, folder class only).

Kept for backwards compatibility; it now delegates to auto_label.py, which writes to a separate folder with a
provenance record instead of into the human label folders. Equivalent to:

    python dataset/auto_label.py --conf 0.05 --augment --folder-class-only [--apply]

Note: confidence 0.05 with TTA produces many wrong boxes; these labels need human review before any use.
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    cmd = [sys.executable, str(Path(__file__).with_name("auto_label.py")), "--conf", "0.05", "--augment",
           "--folder-class-only", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))
