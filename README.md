# Staged Rule Acquisition in WishGLD

**Dissociation and Compression Across Learning**

Analysis code for the Master's thesis (University of Helsinki, Neuroscience, 2026)
by Jianlun Sun, supervised by Benjamin Ultan Cowley.

---

## Overview

This repository contains the behavioral, EEG, and individual-difference analysis
code used in the thesis. The study reframes rule acquisition in the
**Wisconsin-ish Global–Local Dissociation (WishGLD)** task — a WCST-like
set-shifting paradigm — as a *temporally extended, multi-marker transition*
rather than a single learning event.

Three acquisition markers are estimated per block and compared:

- **First-correct** — the earliest observable behavioral success.
- **Logical collapse** — the trial at which accumulated feedback reduces the
  rule hypothesis space to a single consistent candidate.
- **HMM-core acquisition** — behavioral stabilization, estimated from a
  two-state (search / acquired) hidden Markov model fit to trial-wise responses.

Feedback-locked EEG (**FRN** and **P3b**) is analyzed around these markers with
hierarchical Bayesian stage contrasts, and inter-subject representational
similarity analysis (**IS-RSA**) tests whether participants with similar
behavioral acquisition trajectories also show similar EEG transition patterns.

## Repository structure

| Folder | Contents |
|---|---|
| `Behavior/` | Log parsing, trial-wise reconstruction, HMM long tables, logical-collapse candidate tracking, missing-correctness audit, and behavioral analyses. |
| `eeg_tables/` | EEG/behavior event alignment, feedback- and cue-locked ERP table construction (MATLAB + Python), ERP component extraction, and main-text figure generation. |
| `solidity/` | Core acquisition/transition analyses: unified block-level HMM dynamics, QC gate, ERP stage contrasts, FRN/P3b change analyses, global-precedence figures, and HMM diagnostics. |
| `expand/` | Robustness and extension analyses: hierarchical Bayesian stage and two-step models, trial-count sensitivity, marker-validity checks, and IS-RSA (trajectory RDMs, sensitivity, plots). |
| `config/` | Lightweight acquisition/preprocessing references: CTAP config and actiCAP 32-channel coordinates. |

External toolboxes and environments (EEGLAB, CTAP, Python virtual environments)
are **not** included; only project-specific scripts and lightweight config files
are tracked here.

## Methods → code map

| Thesis section | Key scripts |
|---|---|
| 2.5 HMM transition markers | `Behavior/HMM` core + `solidity/hmm_unified_block_dynamics.py` |
| 2.6 Logical collapse marker | `Behavior/trait_extractor.py`, `Behavior/hmm_long_tables.py` |
| 2.7 Behavioral confirmatory analyses | `eeg_tables/generate_main_text_figures_1s.py` (paired *t*-tests, BH-FDR families) |
| 2.8.1–2.8.2 EEG preprocessing & ERP features | `eeg_tables/build_eeg_tables_feedback_locked.m` (primary), `..._py.py` |
| 2.8.3 Trial retention & cell balance | `solidity/run_eeg_qc_gate_hmm_core_epochs.py` |
| 2.8.4 ERP stage-contrast inference | `solidity/eeg_acquisition_sequence_stages.py`, `expand/run_hierarchical_bayesian_stage_twostep.py` |
| 3.3 Two-step acquisition dynamics | `expand/run_two_step_transition_confirmation.py`, `..._stage_twostep.py` |
| 2.9 IS-RSA | `expand/run_isrsa_two_blocklevel_rdms.py`, `expand/run_isrsa_blocklevel_sensitivity.py` |

## Dependencies

- **Python** 3.10+ — `numpy`, `pandas`, `scipy`, `matplotlib`, `mne`,
  `pymc`, `arviz` (Bayesian stage models and IS-RSA).
- **MATLAB** with **EEGLAB** — feedback/cue-locked ERP table construction and
  event alignment.
- **EEG preprocessing** was performed with **CTAP** (config in `config/`);
  the preprocessing pipeline itself is not redistributed here.

## Reproducing the analyses

The scripts assume preprocessed EEG `.set` files and raw behavioral logs as
inputs; **raw data are not included** in this repository (see *Data availability*).
A typical order is:

1. **Behavior** — parse logs → trial-wise tables → HMM block/trial metrics
   (`Behavior/`).
2. **EEG tables** — align behavior to EEG events and build feedback-locked ERP
   tables (`eeg_tables/`).
3. **Core analyses** — QC gate, ERP stage contrasts, and acquisition-marker
   comparisons (`solidity/`).
4. **Extensions** — hierarchical Bayesian models, two-step contrasts, and IS-RSA
   (`expand/`).

> **Note on paths.** Several scripts use absolute input/output paths set at the
> top of each file; adjust these to your local environment before running.

## Data availability

Behavioral and EEG data are not shared in this repository. Data were collected
prior to GDPR (2018) and before current open-science EEG data-sharing norms;
EEG additionally carries participant re-identification risk. Requests regarding
data should be directed to the thesis author and supervisor.

## Citation

Sun, J. (2026). *Staged Rule Acquisition in WishGLD: Dissociation and
Compression Across Learning* [Master's thesis, University of Helsinki].
E-Thesis / HELDA – Digital Repository of the University of Helsinki.
