# SWORD v2 — Research Paper Materials & Artifacts Hub

This folder contains all scientific figures, publication-ready LaTeX tables, raw empirical JSON data, and markdown comparison reports for the **SWORD v2: Small-Weapon Object Recognition and Detection via SOTA Multi-Model Consensus Ensembles** paper.

---

## Folder Structure

```
PAPER/
├── figures/
│   ├── benchmark_comparison/
│   │   ├── fig_benchmark_comparison_1_vs_2.png (.pdf)  # Macro comparison (mAP@50, mAP@50:95, P, R, F1)
│   │   ├── fig_benchmark_comparison_1.png (.pdf)        # Run 1 Standalone vs Ensemble
│   │   └── fig_benchmark_comparison_2.png (.pdf)        # Run 2 Standalone vs Optimized Ensemble
│   ├── per_class_analysis/
│   │   ├── fig_per_class_delta.png (.pdf)              # Net AP@50 gain per class
│   │   └── fig_per_class_f1_1.png (.pdf)               # Per-class F1 distribution
│   ├── scale_sensitivity_sowdet/
│   │   ├── fig_scale_sensitivity_1_vs_2.png (.pdf)     # Small, Medium, Large object scale breakdown
│   │   └── fig_scale_sensitivity_1.png (.pdf)          # Baseline scale sensitivity
│   └── false_alarm_calibration/
│       └── fig_far_vs_confidence_1.png (.pdf)          # False Discovery Rate (FDR) vs confidence sweep
├── latex_tables/
│   ├── table_comparison_macro.tex                      # LaTeX table: Macro metrics comparison
│   ├── table_comparison_per_class.tex                  # LaTeX table: Per-class AP@50 comparison
│   ├── table_comparison_scale.tex                      # LaTeX table: Scale sensitivity comparison
│   ├── table_main_benchmark_1.tex                      # LaTeX table: Benchmark Results 1
│   ├── table_main_benchmark_2.tex                      # LaTeX table: Benchmark Results 2
│   ├── table_per_class_breakdown_1.tex                 # LaTeX table: Per-class breakdown (Run 1)
│   ├── table_per_class_breakdown_2.tex                 # LaTeX table: Per-class breakdown (Run 2)
│   ├── table_scale_sensitivity_1.tex                   # LaTeX table: Scale breakdown (Run 1)
│   └── table_scale_sensitivity_2.tex                   # LaTeX table: Scale breakdown (Run 2)
├── raw_results_json/
│   ├── benchmark_results_1.json                        # Full telemetry for Benchmark 1
│   ├── benchmark_results_2.json                        # Full telemetry for Benchmark 2
│   ├── optimized_parameters.json                       # Discovered non-training weights and thresholds
│   └── sota_benchmark_full_report.json                 # Initial calibration sweep report
└── reports_and_summaries/
    └── comparison_summary_report.md                    # Detailed executive comparison report
```

---

## Empirical Highlights

- **Overall mAP@50**: **0.8338 → 0.8518** (+2.16% relative gain; beats best standalone RT-DETR-L 0.8472)
- **Overall mAP@50:95**: **0.5892 → 0.6126** (+3.97% relative gain)
- **Overall Precision**: **0.8970 → 0.9259** (+3.22% relative gain)
- **Overall F1-Score**: **0.8512 → 0.8650** (+1.62% relative gain)
- **Medium Object AP ($32^2 - 96^2$ px)**: **0.4646 → 0.5253** (+13.08% relative gain)
- **Zero Retraining Required**: 100% achieved via class-adaptive WBF weighting, consensus-boosted confidence scaling, and per-class thresholding.
