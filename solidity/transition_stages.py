"""Single source of truth for trial-level transition-stage classification.

Every ERP stage analysis classifies each feedback trial relative to the block's
first-correct trial (fc) and HMM core-acquisition trial (core). That logic used
to be hand-written in four places with quietly divergent conventions; it now
lives here so the category semantics are declared ONCE and any change is made in
a single spot.

Boundary conventions (canonical reference -- cite this docstring in methods):
  Given a per-trial index `trial_col`, the block-level `first_correct_trial`
  (fc) and `acquisition_trial_core` (core), and per-trial `correctness`:

    search_error              : trial <  fc   and correctness == 0
    search_correct            : trial <  fc   and correctness == 1
    transition_first_correct  : trial == fc
    transition_pre_core       : fc < trial < core        (correctness filter optional)
    acquired                  : trial >= core (or == core) (correctness filter optional)

Index convention (this is the subtle one -- read it):
  fc and core are produced by the unified HMM pipeline in `trial_index_1based`
  units. Passing trial_col="trial_id" is only equivalent when
  trial_id == trial_index_1based for every block (i.e. no trials were dropped in
  HMM sequence preparation). `eeg_acquisition_sequence_stages` historically uses
  trial_id; that behavior is preserved here, not silently changed.

The variants each caller selects (kept byte-for-byte identical to prior behavior;
verified by test_transition_stages.py):
  eeg_acquisition_sequence_stages       : trial_id, acquired '==' core, NO
      correctness filter on pre_core/acquired, bare names, OVERLAPPING views
      (a fc==core trial is in both transition_first_correct and acquired) ->
      consumes stage_masks() directly.
  p300_frn_changes_and_global_precedence: trial_index_1based, acquired '>=' core,
      correctness filter, '_correct' names, WITH search_correct, acquired-wins.
  search_to_transition_erp_contrasts    : trial_index_1based, correctness filter,
      only {search_error, transition_pre_core_correct, transition_first_correct}.
  run_bayesian_stage_and_twostep_sensitivity (_build_p300_stage_cells):
      trial_index_1based, acquired '>=' core, correctness filter, '_correct'
      names, no search_correct, acquired-wins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _scalar_or_col(df: pd.DataFrame, value, col: str) -> pd.Series:
    """fc/core may be passed as a per-block scalar or read from a column."""
    if value is None:
        return _num(df, col)
    return pd.Series(float(value), index=df.index)


# Internal mask keys (stable) -> default output labels.
_BARE_NAMES = {
    "search_error": "search_error",
    "search_correct": "search_correct",
    "transition_first_correct": "transition_first_correct",
    "transition_pre_core": "transition_pre_core",
    "acquired": "acquired_core",
}
_CORRECT_NAMES = {
    **_BARE_NAMES,
    "transition_pre_core": "transition_pre_core_correct",
    "acquired": "acquired_correct",
}

_DEFAULT_ORDER = (
    "search_error",
    "search_correct",
    "transition_first_correct",
    "transition_pre_core",
    "acquired",
)


def stage_masks(
    df: pd.DataFrame,
    *,
    trial_col: str = "trial_index_1based",
    fc=None,
    core=None,
    acquired_boundary: str = ">=",
    pre_core_require_correct: bool = True,
    acquired_require_correct: bool = True,
) -> dict:
    """Boolean masks (aligned to df.index) for every stage.

    fc / core may be given as per-block scalars or read from the
    'first_correct_trial' / 'acquisition_trial_core' columns when left None.
    Callers that build OVERLAPPING category views (e.g.
    eeg_acquisition_sequence_stages, where a trial at fc==core belongs to both
    transition_first_correct and acquired) consume these masks directly rather
    than collapsing to one label per row.
    """
    t = _num(df, trial_col)
    fcv = _scalar_or_col(df, fc, "first_correct_trial")
    corev = _scalar_or_col(df, core, "acquisition_trial_core")
    corr = _num(df, "correctness")

    if acquired_boundary == ">=":
        acq = t >= corev
    elif acquired_boundary == "==":
        acq = t == corev
    else:
        raise ValueError(f"acquired_boundary must be '>=' or '==', got {acquired_boundary!r}")

    pre = (t > fcv) & (t < corev)
    if pre_core_require_correct:
        pre = pre & (corr == 1)
    if acquired_require_correct:
        acq = acq & (corr == 1)

    return {
        "search_error": (t < fcv) & (corr == 0),
        "search_correct": (t < fcv) & (corr == 1),
        "transition_first_correct": t == fcv,
        "transition_pre_core": pre,
        "acquired": acq,
    }


def assign_transition_stage(
    df: pd.DataFrame,
    *,
    trial_col: str = "trial_index_1based",
    fc=None,
    core=None,
    acquired_boundary: str = ">=",
    pre_core_require_correct: bool = True,
    acquired_require_correct: bool = True,
    include_search_correct: bool = False,
    correct_suffix: bool = True,
    order: tuple = _DEFAULT_ORDER,
) -> pd.Series:
    """One category per row (empty string == unclassified).

    Overlapping trials resolve by last-write-wins following `order`; this is how
    a fc==core trial is labeled. The default order lets 'acquired' win over
    'transition_first_correct' at fc==core, matching the P300 / Bayesian callers.
    Pass an `order` that omits a stage to skip it entirely (e.g. the
    search_to_transition caller omits 'acquired' and 'search_correct').
    """
    masks = stage_masks(
        df,
        trial_col=trial_col,
        fc=fc,
        core=core,
        acquired_boundary=acquired_boundary,
        pre_core_require_correct=pre_core_require_correct,
        acquired_require_correct=acquired_require_correct,
    )
    names = _CORRECT_NAMES if correct_suffix else _BARE_NAMES
    cat = pd.Series("", index=df.index, dtype="object")
    for key in order:
        if key == "search_correct" and not include_search_correct:
            continue
        cat[masks[key]] = names[key]
    return cat
