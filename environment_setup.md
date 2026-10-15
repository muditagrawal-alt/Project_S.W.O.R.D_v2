# S.W.O.R.D. Phase 4 - Environment Setup Guide

## 1. System Requirements & Compatibility
- **OS**: Windows 10/11 (64-bit)
- **Target Python Version**: **Python 3.11** (Recommended: 3.11.9, located at `C:\Users\nidhi\AppData\Local\Programs\Python\Python311\python.exe`)
- **Important**: Do not use Python 3.14 for OpenMMLab / PyTorch as binary wheels for MMCV/PyTorch are not available.
- **NumPy Warning**: Always use `numpy<2.0.0` (e.g. `numpy==1.26.4`) to prevent ABI conflicts with pre-compiled PyTorch 2.1/2.2 and MMCV wheels.

---

## 2. Virtual Environment Creation

Open PowerShell and execute:

```powershell
# Navigate to Phase 4 directory
cd "d:\New folder\OneDrive\IILM\IILM 4th Yr\Phase4\phase4_action"

# Create a dedicated Python 3.11 virtual environment
& "C:\Users\nidhi\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1

# Upgrade pip and wheel
python -m pip install --upgrade pip wheel setuptools
```

---

## 3. PyTorch Installation

### For CPU-only:
```powershell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

### For NVIDIA GPU (CUDA 11.8):
```powershell
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### For NVIDIA GPU (CUDA 12.1):
```powershell
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Install OpenMMLab Core (MMEngine, MMCV, MMPose, MMAction2)

Using OpenMMLab's package manager `openmim` guarantees prebuilt Windows wheels without needing MSVC C++ compilation:

```powershell
pip install -U openmim
mim install mmengine==0.10.3

# Install prebuilt MMCV 2.1.0 (matching PyTorch 2.1)
# For CPU:
mim install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

# OR For CUDA 11.8:
mim install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# Install MMPose and MMAction2
mim install "mmpose>=1.3.0"
mim install "mmaction2>=1.2.0"
```

---

## 5. Install Stand-in Tracker/Pose & Auxiliary Libraries

```powershell
pip install ultralytics lapx opencv-python pandas scikit-learn pyyaml tqdm matplotlib
```

---

## 6. Verification Command

Run the verification test script:
```powershell
python -c "
import torch, cv2, yaml, sklearn
print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
try:
    import mmengine, mmcv, mmpose, mmaction
    print('MMAction2:', mmaction.__version__, '| MMPose:', mmpose.__version__, '| MMCV:', mmcv.__version__)
except ImportError as e:
    print('Note: MMAction2 official package not loaded; standalone PyTorch fallback will be used:', e)
"
```
