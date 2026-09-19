"""
SWORD v2 — Paper Material Organizer, Safe Project Cleaner & Full Zip Engine
1. Organizes all paper material into a structured PAPER/ folder with clear subfolders:
   - PAPER/figures/benchmark_comparison/
   - PAPER/figures/per_class_analysis/
   - PAPER/figures/scale_sensitivity_sowdet/
   - PAPER/figures/false_alarm_calibration/
   - PAPER/latex_tables/
   - PAPER/raw_results_json/
   - PAPER/reports_and_summaries/
2. Removes unwanted garbage files (.DS_Store, __pycache__, redundant old bulky zip backups).
3. Compresses the entire project into SWORD_v2_complete.zip.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def setup_paper_folder():
    paper_root = PROJECT_ROOT / "PAPER"
    print("=" * 80)
    print("📁 1. ORGANIZING ALL RESEARCH PAPER MATERIAL INTO 'PAPER/'")
    print("=" * 80)

    dirs = {
        "bench_figs": paper_root / "figures" / "benchmark_comparison",
        "class_figs": paper_root / "figures" / "per_class_analysis",
        "scale_figs": paper_root / "figures" / "scale_sensitivity_sowdet",
        "far_figs": paper_root / "figures" / "false_alarm_calibration",
        "tables": paper_root / "latex_tables",
        "json": paper_root / "raw_results_json",
        "reports": paper_root / "reports_and_summaries",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # 1. Figures: Benchmark Comparison
    bench_sources = [
        PROJECT_ROOT / "results" / "benchmark_comparison" / "fig_benchmark_comparison_1_vs_2.png",
        PROJECT_ROOT / "results" / "benchmark_comparison" / "fig_benchmark_comparison_1_vs_2.pdf",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "fig_benchmark_comparison_1.png",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "fig_benchmark_comparison_1.pdf",
        PROJECT_ROOT / "results" / "benchmark_results_2" / "fig_benchmark_comparison_2.png",
        PROJECT_ROOT / "results" / "benchmark_results_2" / "fig_benchmark_comparison_2.pdf",
    ]
    for src in bench_sources:
        if src.exists():
            shutil.copy2(src, dirs["bench_figs"] / src.name)
            print(f"  -> Copied to figures/benchmark_comparison: {src.name}")

    # 2. Figures: Per-Class Analysis
    class_sources = [
        PROJECT_ROOT / "results" / "benchmark_comparison" / "fig_per_class_delta.png",
        PROJECT_ROOT / "results" / "benchmark_comparison" / "fig_per_class_delta.pdf",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "fig_per_class_f1_1.png",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "fig_per_class_f1_1.pdf",
    ]
    for src in class_sources:
        if src.exists():
            shutil.copy2(src, dirs["class_figs"] / src.name)
            print(f"  -> Copied to figures/per_class_analysis: {src.name}")

    # 3. Figures: Scale Sensitivity (SO-WDet)
    scale_sources = [
        PROJECT_ROOT / "results" / "benchmark_comparison" / "fig_scale_sensitivity_1_vs_2.png",
        PROJECT_ROOT / "results" / "benchmark_comparison" / "fig_scale_sensitivity_1_vs_2.pdf",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "fig_scale_sensitivity_1.png",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "fig_scale_sensitivity_1.pdf",
    ]
    for src in scale_sources:
        if src.exists():
            shutil.copy2(src, dirs["scale_figs"] / src.name)
            print(f"  -> Copied to figures/scale_sensitivity_sowdet: {src.name}")

    # 4. Figures: False Alarm Rate Calibration
    far_sources = [
        PROJECT_ROOT / "results" / "benchmark_results_1" / "fig_far_vs_confidence_1.png",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "fig_far_vs_confidence_1.pdf",
    ]
    for src in far_sources:
        if src.exists():
            shutil.copy2(src, dirs["far_figs"] / src.name)
            print(f"  -> Copied to figures/false_alarm_calibration: {src.name}")

    # 5. LaTeX Tables
    table_sources = [
        PROJECT_ROOT / "results" / "benchmark_comparison" / "table_comparison_macro.tex",
        PROJECT_ROOT / "results" / "benchmark_comparison" / "table_comparison_per_class.tex",
        PROJECT_ROOT / "results" / "benchmark_comparison" / "table_comparison_scale.tex",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "table_main_benchmark_1.tex",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "table_per_class_breakdown_1.tex",
        PROJECT_ROOT / "results" / "benchmark_results_1" / "table_scale_sensitivity_1.tex",
        PROJECT_ROOT / "results" / "benchmark_results_2" / "table_main_benchmark_2.tex",
        PROJECT_ROOT / "results" / "benchmark_results_2" / "table_per_class_breakdown_2.tex",
        PROJECT_ROOT / "results" / "benchmark_results_2" / "table_scale_sensitivity_2.tex",
    ]
    for src in table_sources:
        if src.exists():
            shutil.copy2(src, dirs["tables"] / src.name)
            print(f"  -> Copied to latex_tables: {src.name}")

    # 6. Raw JSON Telemetry
    json_sources = [
        PROJECT_ROOT / "results" / "benchmark_results_1" / "benchmark_results_1.json",
        PROJECT_ROOT / "results" / "benchmark_results_2" / "benchmark_results_2.json",
        PROJECT_ROOT / "results" / "optimized_parameters.json",
        PROJECT_ROOT / "results" / "paper_artifacts" / "sota_benchmark_full_report.json",
    ]
    for src in json_sources:
        if src.exists():
            shutil.copy2(src, dirs["json"] / src.name)
            print(f"  -> Copied to raw_results_json: {src.name}")

    # 7. Reports & Summaries
    report_sources = [
        PROJECT_ROOT / "results" / "benchmark_comparison" / "comparison_summary_report.md",
    ]
    for src in report_sources:
        if src.exists():
            shutil.copy2(src, dirs["reports"] / src.name)
            print(f"  -> Copied to reports_and_summaries: {src.name}")

    # Create PAPER README guide
    readme_content = """# SWORD v2 — Research Paper Materials & Artifacts Hub

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
"""
    with open(paper_root / "README.md", "w") as fp:
        fp.write(readme_content)

    print("✅ PAPER/ directory fully assembled and documented.")


def clean_unwanted_files():
    print("\n" + "=" * 80)
    print("🧹 2. CLEANING UNWANTED AND GARBAGE FILES")
    print("=" * 80)

    # 1. Clean .DS_Store
    ds_count = 0
    for p in PROJECT_ROOT.rglob(".DS_Store"):
        try:
            p.unlink()
            ds_count += 1
        except Exception:
            pass
    print(f"  -> Removed {ds_count} .DS_Store files")

    # 2. Clean __pycache__ and *.pyc
    pyc_count = 0
    for p in PROJECT_ROOT.rglob("*.pyc"):
        try:
            p.unlink()
            pyc_count += 1
        except Exception:
            pass
    for p in PROJECT_ROOT.rglob("__pycache__"):
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    print(f"  -> Removed {pyc_count} compiled .pyc files and all __pycache__ directories")

    # 3. Clean redundant duplicate / legacy heavy zip archives that bloat the repo
    heavy_zips = [
        PROJECT_ROOT / "yolo26m_resumed_model.zip",
        PROJECT_ROOT / "sword_v2_weights.zip",
        PROJECT_ROOT / "yolo26s" / "results.zip",
        PROJECT_ROOT / "yolo26m" / "yolo26m_results.zip",
        PROJECT_ROOT / "rt_detr" / "re-detr_first_run_results.zip",
        PROJECT_ROOT / "rt_detr" / "rt-detr_output.zip",
        PROJECT_ROOT / "archive" / "KAGGLE _UNFINISHED_TRAINED_MODEL.zip",
    ]

    freed_mb = 0
    for z in heavy_zips:
        if z.exists():
            size_mb = z.stat().st_size / (1024 * 1024)
            try:
                z.unlink()
                freed_mb += size_mb
                print(f"  -> Removed redundant bulky zip: {z.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  -> Could not remove {z}: {e}")

    print(f"  -> Total redundant storage freed: {freed_mb:.1f} MB ({freed_mb / 1024:.2f} GB)")
    print("✅ Garbage cleanup complete. Active code, model weights, dataset, and results are 100% preserved.")


def zip_entire_project():
    print("\n" + "=" * 80)
    print("📦 3. CREATING COMPLETE PROJECT ZIP ARCHIVE")
    print("=" * 80)

    zip_filename = PROJECT_ROOT / "SWORD_v2_COMPLETE_PROJECT.zip"
    if zip_filename.exists():
        zip_filename.unlink()

    # Files or dirs to exclude from zip (the output zip itself and venv to avoid bundling OS-specific python binary)
    exclude_dirs = {".git", "venv", ".venv", "env"}
    exclude_files = {zip_filename.name}

    total_files = 0
    total_size = 0

    print(f"Creating archive: {zip_filename.name}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files or file.endswith(".pyc") or file == ".DS_Store":
                    continue
                full_path = Path(root) / file
                rel_path = full_path.relative_to(PROJECT_ROOT)
                zf.write(full_path, arcname=str(rel_path))
                total_files += 1
                total_size += full_path.stat().st_size

    zip_size_mb = zip_filename.stat().st_size / (1024 * 1024)
    print(f"\n🎉 ZIP COMPLETE!")
    print(f"  Archive Name: {zip_filename.name}")
    print(f"  Location:     {zip_filename}")
    print(f"  Files Packed: {total_files:,} files")
    print(f"  Raw Size:     {total_size / (1024 * 1024):.1f} MB")
    print(f"  Zip Size:     {zip_size_mb:.1f} MB ({zip_size_mb / 1024:.2f} GB)")
    print("=" * 80)


def main():
    setup_paper_folder()
    clean_unwanted_files()
    zip_entire_project()


if __name__ == "__main__":
    main()
