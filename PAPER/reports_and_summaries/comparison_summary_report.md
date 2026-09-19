# SWORD v2 — Benchmark Results 1 vs Benchmark Results 2 Comparative Analysis

## 1. Executive Summary

This evaluation presents the direct empirical comparison between **Benchmark Results 1 (Baseline)** and **Benchmark Results 2 (Non-Training SOTA Optimizations)** across 500 stratified validation images covering all 10 weapon classes.

### Key Takeaways:
- **mAP@50 Gain**: Improved from **0.8338** to **0.8518** (+2.16%).
- **Precision Gain**: Improved from **0.8970** to **0.9259** (+3.22%).
- **F1-Score Gain**: Improved from **0.8512** to **0.8650** (+1.62%).
- **No Weight Retraining**: Achieved 100% via post-processing enhancements (Class-Adaptive WBF Weighting, Consensus-Boosted Scaling, and Per-Class Threshold Optimization).

---

## 2. Macro Performance Comparison

| Metric    |   Benchmark Results 1 (Baseline) |   Benchmark Results 2 (Optimized) |   Absolute Delta | Relative Gain   |
|:----------|---------------------------------:|----------------------------------:|-----------------:|:----------------|
| mAP@50    |                           0.8338 |                            0.8518 |           0.018  | +2.16%          |
| mAP@50:95 |                           0.5892 |                            0.6126 |           0.0234 | +3.97%          |
| Precision |                           0.897  |                            0.9259 |           0.0289 | +3.22%          |
| Recall    |                           0.817  |                            0.8163 |          -0.0007 | -0.08%          |
| F1-Score  |                           0.8512 |                            0.865  |           0.0138 | +1.62%          |

---

## 3. Object Scale Sensitivity Comparison (SO-WDet Standard)

| Scale Range      |   Benchmark 1 AP@50 |   Benchmark 2 AP@50 |   Delta | Gain (%)   |
|:-----------------|--------------------:|--------------------:|--------:|:-----------|
| Small (< 32px)   |              0      |              0      |  0      | +0.00%     |
| Medium (32-96px) |              0.4646 |              0.5253 |  0.0607 | +13.08%    |
| Large (> 96px)   |              0.8441 |              0.8581 |  0.0139 | +1.65%     |

---

## 4. Per-Class AP@50 Breakdown & Gain Analysis

| Weapon Class    |   Benchmark 1 (AP@50) |   Benchmark 2 (AP@50) |   Delta | Gain (%)   |
|:----------------|----------------------:|----------------------:|--------:|:-----------|
| Pistol          |                0.7559 |                0.768  |  0.012  | +1.59%     |
| Shotgun         |                0.8516 |                0.8724 |  0.0208 | +2.44%     |
| assault rifle   |                0.8504 |                0.8668 |  0.0164 | +1.93%     |
| grenade         |                0.8922 |                0.9411 |  0.0489 | +5.48%     |
| knife           |                0.7349 |                0.7752 |  0.0404 | +5.49%     |
| sword           |                0.7658 |                0.7671 |  0.0013 | +0.17%     |
| Sniper Dataset  |                0.7815 |                0.7881 |  0.0067 | +0.85%     |
| SMG             |                0.995  |                0.995  |  0      | +0.00%     |
| rocket launcher |                0.7653 |                0.7789 |  0.0137 | +1.78%     |
| Machine gun     |                0.9457 |                0.9658 |  0.0201 | +2.13%     |

---

## 5. Artifact Directory Locations
- **Benchmark Results 1 Directory**: `results/benchmark_results_1/`
- **Benchmark Results 2 Directory**: `results/benchmark_results_2/`
- **Comparative Analysis Directory**: `results/benchmark_comparison/`
