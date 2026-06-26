"""Equivalence tests for transition_stages.py.

Each test re-implements the ORIGINAL hand-written classification of one analysis
file (the "before" code) and asserts the centralized function reproduces it
exactly on synthetic trial tables -- including the tricky cases: fc==core
(immediate acquisition, ~85% of blocks), incorrect trials inside the pre-core
window, and the overlapping-frame double-count in eeg_acquisition_sequence_stages.

Run:  python3 test_transition_stages.py    (prints OK or raises AssertionError)
Pure numpy/pandas -- no project data required.
"""
import numpy as np
import pandas as pd

from transition_stages import assign_transition_stage, stage_masks


def _synthetic(seed: int = 0) -> pd.DataFrame:
    """A block-structured trial table spanning all the edge cases."""
    rng = np.random.default_rng(seed)
    rows = []
    # (fc, core) pairs: lag>0, immediate (fc==core), and a long transition.
    block_specs = [(3, 6), (4, 4), (2, 2), (1, 5), (5, 9), (2, 8)]
    for bi, (fc, core) in enumerate(block_specs):
        n = core + 4
        for t in range(1, n + 1):
            corr = int(rng.integers(0, 2))
            # guarantee the fc trial is correct (true by construction in real data)
            if t == fc:
                corr = 1
            rows.append(
                {
                    "subj": "s01",
                    "block_id": bi,
                    "trial_id": t,
                    "trial_index_1based": t,  # equal here; divergence is a data property
                    "first_correct_trial": fc,
                    "acquisition_trial_core": core,
                    "correctness": corr,
                }
            )
    return pd.DataFrame(rows)


def _orig_p300(m: pd.DataFrame) -> pd.Series:
    """Original inline logic of p300_frn_changes_and_global_precedence.py."""
    is_fc = m["trial_index_1based"] == m["first_correct_trial"]
    is_precore = (
        (m["trial_index_1based"] > m["first_correct_trial"])
        & (m["trial_index_1based"] < m["acquisition_trial_core"])
        & (m["correctness"] == 1)
    )
    is_sw = (m["trial_index_1based"] < m["first_correct_trial"]) & (m["correctness"] == 0)
    is_sc = (m["trial_index_1based"] < m["first_correct_trial"]) & (m["correctness"] == 1)
    is_acq = (m["trial_index_1based"] >= m["acquisition_trial_core"]) & (m["correctness"] == 1)
    cat = pd.Series("", index=m.index, dtype="object")
    cat[is_sw] = "search_error"
    cat[is_sc] = "search_correct"
    cat[is_fc] = "transition_first_correct"
    cat[is_precore] = "transition_pre_core_correct"
    cat[is_acq] = "acquired_correct"
    return cat


def _orig_search(m: pd.DataFrame) -> pd.Series:
    """Original inline logic of search_to_transition_erp_contrasts.py."""
    CATEGORIES = ["search_error", "transition_pre_core_correct", "transition_first_correct"]
    is_fc = m["trial_index_1based"] == m["first_correct_trial"]
    is_precore = (
        (m["trial_index_1based"] > m["first_correct_trial"])
        & (m["trial_index_1based"] < m["acquisition_trial_core"])
        & (m["correctness"] == 1)
    )
    is_sw = (m["trial_index_1based"] < m["first_correct_trial"]) & (m["correctness"] == 0)
    cat = pd.Series("", index=m.index, dtype="object")
    cat[is_sw] = "search_error"
    cat[is_precore] = "transition_pre_core_correct"
    cat[is_fc] = "transition_first_correct"
    cat[~cat.isin(CATEGORIES)] = ""  # the .isin filter keeps only these three
    return cat


def _orig_bayes_p300(m: pd.DataFrame) -> pd.Series:
    """Original inline logic of run_bayesian..._sensitivity._build_p300_stage_cells."""
    is_fc = m["trial_index_1based"] == m["first_correct_trial"]
    is_precore = (
        (m["trial_index_1based"] > m["first_correct_trial"])
        & (m["trial_index_1based"] < m["acquisition_trial_core"])
        & (m["correctness"] == 1)
    )
    is_sw = (m["trial_index_1based"] < m["first_correct_trial"]) & (m["correctness"] == 0)
    is_acq = (m["trial_index_1based"] >= m["acquisition_trial_core"]) & (m["correctness"] == 1)
    cat = pd.Series("", index=m.index, dtype="object")
    cat[is_sw] = "search_error"
    cat[is_fc] = "transition_first_correct"
    cat[is_precore] = "transition_pre_core_correct"
    cat[is_acq] = "acquired_correct"
    return cat


def test_p300_profile():
    m = _synthetic(1)
    got = assign_transition_stage(
        m,
        trial_col="trial_index_1based",
        acquired_boundary=">=",
        include_search_correct=True,
        correct_suffix=True,
        order=("search_error", "search_correct", "transition_first_correct",
               "transition_pre_core", "acquired"),
    )
    pd.testing.assert_series_equal(got, _orig_p300(m), check_names=False)


def test_search_profile():
    m = _synthetic(2)
    got = assign_transition_stage(
        m,
        trial_col="trial_index_1based",
        correct_suffix=True,
        order=("search_error", "transition_pre_core", "transition_first_correct"),
    )
    pd.testing.assert_series_equal(got, _orig_search(m), check_names=False)


def test_bayes_p300_profile():
    m = _synthetic(3)
    got = assign_transition_stage(
        m,
        trial_col="trial_index_1based",
        acquired_boundary=">=",
        include_search_correct=False,
        correct_suffix=True,
        order=("search_error", "transition_first_correct", "transition_pre_core", "acquired"),
    )
    pd.testing.assert_series_equal(got, _orig_bayes_p300(m), check_names=False)


def test_acquisition_sequence_overlapping_masks():
    """eeg_acquisition_sequence_stages: trial_id, ==core, no correctness filter,
    and a fc==core trial must appear in BOTH transition_first_correct and acquired."""
    m = _synthetic(4)
    for (subj, bid), b in m.groupby(["subj", "block_id"]):
        fc = int(b["first_correct_trial"].iloc[0])
        core = int(b["acquisition_trial_core"].iloc[0])
        t = pd.to_numeric(b["trial_id"])
        corr = pd.to_numeric(b["correctness"])
        # original frames
        orig = {
            "search_error": b[(t < fc) & (corr == 0)],
            "transition_first_correct": b[t == fc],
            "transition_pre_core": b[(t > fc) & (t < core)],
            "acquired_core": b[t == core],
        }
        masks = stage_masks(
            b, trial_col="trial_id", fc=fc, core=core,
            acquired_boundary="==", pre_core_require_correct=False,
            acquired_require_correct=False,
        )
        got = {
            "search_error": b[masks["search_error"]],
            "transition_first_correct": b[masks["transition_first_correct"]],
            "transition_pre_core": b[masks["transition_pre_core"]],
            "acquired_core": b[masks["acquired"]],
        }
        for k in orig:
            assert list(orig[k].index) == list(got[k].index), f"{bid}:{k} mismatch"
        # explicit double-count guarantee at fc==core
        if fc == core:
            shared = set(got["transition_first_correct"].index) & set(got["acquired_core"].index)
            assert shared, f"block {bid}: fc==core trial should be in both views"


if __name__ == "__main__":
    test_p300_profile()
    test_search_profile()
    test_bayes_p300_profile()
    test_acquisition_sequence_overlapping_masks()
    print("OK: centralized classification reproduces all four original variants")
