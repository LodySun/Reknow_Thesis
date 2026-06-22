import os
import sys
import shutil
import subprocess
from datetime import datetime
import pandas as pd


BASE = "base_dir"
RES_BASE = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results")
SOL = os.path.join(RES_BASE, "solidity")
EXP = os.path.join(RES_BASE, "expand")
BUNDLE_DIR = os.path.join(EXP, "marker_validity_collapse", "full_rerun_bundle")


def _run_script(path: str) -> dict:
    cmd = [sys.executable, path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "script": path,
        "return_code": int(p.returncode),
        "stdout_tail": (p.stdout or "")[-4000:],
        "stderr_tail": (p.stderr or "")[-4000:],
    }


def _copy_if_exists(src: str, dst_dir: str) -> bool:
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        return True
    return False


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)

    # 1) Rerun related analysis/figure scripts
    scripts = [
        os.path.join(BASE, "codes", "EEG", "solidity", "make_mechanism_four_figures.py"),
        os.path.join(BASE, "codes", "EEG", "solidity", "p300_frn_changes_and_global_precedence.py"),
        os.path.join(BASE, "codes", "EEG", "solidity", "plot_global_precedence_acquired_phase.py"),
        os.path.join(BASE, "codes", "EEG", "expand", "plot_global_precedence_overall_and_search.py"),
        os.path.join(BASE, "codes", "EEG", "expand", "plot_hmm_recomputed_search_shift_figure.py"),
        os.path.join(BASE, "codes", "EEG", "expand", "run_two_step_transition_confirmation.py"),
        os.path.join(BASE, "codes", "EEG", "expand", "test_two_step_compression_interaction.py"),
        os.path.join(BASE, "codes", "EEG", "expand", "run_expand_transition_analyses.py"),
        os.path.join(BASE, "codes", "EEG", "expand", "marker_validity_collapse", "run_marker_independence_validation.py"),
        os.path.join(BASE, "codes", "EEG", "expand", "marker_validity_collapse", "run_probabilistic_collapse_sensitivity.py"),
        os.path.join(BASE, "codes", "EEG", "expand", "marker_validity_collapse", "run_hmm_core_sensitivity_grid.py"),
        os.path.join(BASE, "codes", "EEG", "expand", "marker_validity_collapse", "build_probabilistic_marker_outputs.py"),
    ]
    logs = []
    for s in scripts:
        logs.append(_run_script(s))

    run_log = pd.DataFrame(logs)
    run_log_path = os.path.join(BUNDLE_DIR, "rerun_execution_log.csv")
    run_log.to_csv(run_log_path, index=False)

    # 2) Collect figure files (copy to bundle directory)
    figure_files = [
        os.path.join(SOL, "figure20_behavior_lag_width_summary.png"),
        os.path.join(SOL, "figure21_pacq_aligned_early_late_and_exemplars.png"),
        os.path.join(SOL, "figure22_erp_stage_roles_p3b_focus.png"),
        os.path.join(SOL, "figure23_frn_predicts_acquisition_speed.png"),
        os.path.join(SOL, "figure27_p300_frn_stage_changes_early_vs_late.png"),
        os.path.join(SOL, "figure28_global_precedence_acquired_phase.png"),
        os.path.join(SOL, "figure28_global_precedence_overall.png"),
        os.path.join(SOL, "figure28_global_precedence_search.png"),
        os.path.join(SOL, "figure2_behavior_shift_position_hmm_recomputed.png"),
        os.path.join(EXP, "figure_two_step_transition_confirmation.png"),
        os.path.join(EXP, "figure_expand_transition_core_claims.png"),
        os.path.join(EXP, "marker_validity_collapse", "figure_expand_transition_core_claims_probabilistic.png"),
    ]
    copied_rows = []
    for f in figure_files:
        copied_rows.append({"file": f, "copied": _copy_if_exists(f, BUNDLE_DIR)})
    pd.DataFrame(copied_rows).to_csv(os.path.join(BUNDLE_DIR, "copied_figures_manifest.csv"), index=False)

    # 3) Build unified metrics table
    metric_files = [
        os.path.join(SOL, "solidity_figure_metrics.csv"),
        os.path.join(SOL, "figure28_global_precedence_acquired_phase_summary.csv"),
        os.path.join(SOL, "figure28_global_precedence_overall_summary.csv"),
        os.path.join(SOL, "figure28_global_precedence_search_summary.csv"),
        os.path.join(SOL, "hmm_based_search_metrics_within_vs_cross_tests.csv"),
        os.path.join(EXP, "two_step_transition_confirmation_summary.csv"),
        os.path.join(EXP, "two_step_transition_confirmation_tests.csv"),
        os.path.join(EXP, "two_step_compression_interaction_tests.csv"),
        os.path.join(EXP, "expand1_certainty_commitment_dissociation.csv"),
        os.path.join(EXP, "expand1_event_order_proportions.csv"),
        os.path.join(EXP, "expand2_failure_configuration_models.csv"),
        os.path.join(EXP, "expand3_two_step_transition_tests.csv"),
        os.path.join(EXP, "marker_validity_collapse", "marker_independence_models.csv"),
        os.path.join(EXP, "marker_validity_collapse", "collapse_methods_summary.csv"),
        os.path.join(EXP, "marker_validity_collapse", "hmm_core_sensitivity_summary.csv"),
        os.path.join(EXP, "marker_validity_collapse", "hmm_core_sensitivity_key_checks.csv"),
        # probabilistic + noncircular preferred outputs
        os.path.join(EXP, "marker_validity_collapse", "expand1_certainty_commitment_dissociation_probabilistic.csv"),
        os.path.join(EXP, "marker_validity_collapse", "expand1_event_order_proportions_probabilistic.csv"),
        os.path.join(EXP, "marker_validity_collapse", "expand2_marker_independence_models_probabilistic.csv"),
    ]

    frames = []
    for mf in metric_files:
        if not os.path.exists(mf):
            continue
        try:
            t = pd.read_csv(mf)
            t.insert(0, "source_file", os.path.basename(mf))
            # Global analysis version tag for manuscript consistency.
            if "probabilistic" in os.path.basename(mf):
                t.insert(1, "analysis_version", "probabilistic_noncircular_v1")
            else:
                t.insert(1, "analysis_version", "legacy_or_auxiliary")
            frames.append(t)
            shutil.copy2(mf, os.path.join(BUNDLE_DIR, os.path.basename(mf)))
        except Exception:
            continue

    if frames:
        all_metrics = pd.concat(frames, ignore_index=True, sort=False)
    else:
        all_metrics = pd.DataFrame(columns=["source_file", "analysis_version"])
    all_metrics_path = os.path.join(BUNDLE_DIR, "full_related_metrics_table.csv")
    all_metrics.to_csv(all_metrics_path, index=False)

    # 4) Bundle readme
    readme = os.path.join(BUNDLE_DIR, "README_bundle.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("Full related rerun bundle\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
        f.write("Contents:\n")
        f.write("- rerun_execution_log.csv: script execution status and tails\n")
        f.write("- copied_figures_manifest.csv: which figures were copied\n")
        f.write("- full_related_metrics_table.csv: unified metrics table (with analysis_version tags)\n")
        f.write("- copied figure png files and copied per-analysis metric csv files\n")

    print(f"bundle_dir: {BUNDLE_DIR}")
    print(f"saved: {run_log_path}")
    print(f"saved: {all_metrics_path}")
    print(f"saved: {readme}")


if __name__ == "__main__":
    main()
