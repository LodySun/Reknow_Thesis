%% Build target-locked posterior N1 metrics (expand)
% Outputs:
%   1) target_locked_trial_n1_metrics.csv
%   2) target_locked_subject_contrasts.csv
%
% Three requested metrics:
%   M1) posterior N1 amplitude global-local contrast (acquired + correct trials)
%   M2) posterior N1 latency global-local contrast (acquired + correct trials)
%   M3) hemispheric asymmetry (RH-LH and normalized index) in P1/N1 windows
%
% Notes
% - Uses the same alignment strategy as feedback_locked pipeline.
% - Target-locked event code default = "30" (adjust if your dataset differs).
% - The acquired phase is mapped from legacy phase labels {"stable","late-found"}.

clear; clc;

%% Paths
BASE = 'base_dir';
EEG_DIR = 'eeg_dir';
COMP_TAG = "1s_comp";
ALIGN_MATCH_CSV = fullfile(BASE, 'trials_trialwise', COMP_TAG, 'eeg_behavior_trial_match_status.csv');
TRIALWISE_DIR = fullfile(BASE, 'trials_trialwise', 'trialwise');
OUT_DIR = fullfile(BASE, 'trials_trialwise', COMP_TAG, 'eeg_paper_results', 'expand');
if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

%% Params
SKIP_SUBS = ["reknow011","reknow020","reknow023"];
TARGET_CODE = "30";
EPOCH_WIN_SEC = [-0.2, 0.8];
BASELINE_MS = [-200, 0];

% Time windows (seconds)
WIN_P1 = [0.08, 0.13];
WIN_N1 = [0.14, 0.22];

% Posterior ROIs (fallback-friendly)
ROI_POST = ["P7","P8","PO7","PO8","PO9","PO10","O1","O2","Oz","P3","P4","Pz"];
ROI_L_POST = ["P7","PO7","PO9","O1","P3"];
ROI_R_POST = ["P8","PO8","PO10","O2","P4"];

%% Load alignment metadata
alignT = readtable(ALIGN_MATCH_CSV, 'TextType','string');

%% Subject list
setFiles = dir(fullfile(EEG_DIR, 'reknow*_reknow_wcst.set'));
allSubs = strings(numel(setFiles),1);
for i = 1:numel(setFiles)
    allSubs(i) = extractBefore(string(setFiles(i).name), "_reknow_wcst.set");
end
allSubs = unique(allSubs);

%% Holder
trialRowsCell = cell(numel(allSubs),1);
nTrialCells = 0;

for s = 1:numel(allSubs)
    subj = allSubs(s);
    if any(subj == SKIP_SUBS); continue; end
    fprintf('\n=== Target-locked processing: %s ===\n', subj);

    trialwiseCsv = fullfile(TRIALWISE_DIR, subj + "_trialwise.csv");
    if ~exist(trialwiseCsv, 'file')
        fprintf('[WARN] Missing trialwise file: %s\n', trialwiseCsv);
        continue;
    end
    behT = readtable(trialwiseCsv, 'TextType','string');
    if ~all(ismember(["block_id","trial_id","phase"], string(behT.Properties.VariableNames)))
        fprintf('[WARN] trialwise file missing required columns (block_id/trial_id/phase): %s\n', trialwiseCsv);
        continue;
    end
    % Try to find correctness and rule-level columns with safe fallback names.
    corrCol = pick_col(behT.Properties.VariableNames, ["correctness","correct","is_correct"]);
    rlCol = pick_col(behT.Properties.VariableNames, ["rule_level","rulelevel","level"]);
    if corrCol == "" || rlCol == ""
        fprintf('[WARN] trialwise file missing correctness or rule_level column: %s\n', trialwiseCsv);
        continue;
    end

    behT.block = str2double(string(behT.block_id));
    behT.trial = str2double(string(behT.trial_id));
    behT.correctness = str2double(string(behT.(corrCol)));
    behT.rule_level = lower(string(behT.(rlCol)));

    % Build metadata maps by key block_trial
    key = mk_key(behT.block, behT.trial);
    phaseMap = containers.Map(cellstr(key), cellstr(string(behT.phase)));
    corrMap = containers.Map(cellstr(key), num2cell(behT.correctness));
    rlMap = containers.Map(cellstr(key), cellstr(behT.rule_level));

    % Alignment rows (matched only)
    aSub = alignT(alignT.subj == subj & alignT.match_status == "matched", :);
    if isempty(aSub)
        fprintf('[WARN] No matched alignment rows for %s\n', subj);
        continue;
    end
    aSub.block = str2double(string(aSub.block));
    aSub.trial = str2double(string(aSub.trial));
    aSub.eeg_idx = str2double(string(aSub.eeg_idx));
    aSub = aSub(~isnan(aSub.eeg_idx), :);
    if isempty(aSub); continue; end

    % Load EEG
    setName = subj + "_reknow_wcst.set";
    EEG = pop_loadset('filename', char(setName), 'filepath', EEG_DIR);

    % Target events in original EEG.event timeline
    evType = event_type_to_string(EEG.event);
    tgtEventIdx = find(evType == TARGET_CODE);
    nTgt = numel(tgtEventIdx);
    if nTgt == 0
        fprintf('[WARN] No target code %s in %s\n', TARGET_CODE, subj);
        continue;
    end

    aSub.target_event_num = aSub.eeg_idx + 1;
    aSub = aSub(aSub.target_event_num >= 1 & aSub.target_event_num <= nTgt, :);
    if isempty(aSub); continue; end
    [~,ord] = sort(aSub.target_event_num, 'ascend');
    aSub = aSub(ord, :);
    [~,ia] = unique(aSub.target_event_num, 'stable');
    aSub = aSub(ia, :);

    % Epoch target events first then select mapped trials
    EEG_tgt = pop_epoch(EEG, {char(TARGET_CODE)}, EPOCH_WIN_SEC, 'epochinfo', 'yes');
    EEG_tgt = pop_rmbase(EEG_tgt, BASELINE_MS);
    keepEpoch = aSub.target_event_num;
    keepEpoch = keepEpoch(keepEpoch <= EEG_tgt.trials);
    if isempty(keepEpoch)
        fprintf('[WARN] No target epochs left for %s\n', subj);
        continue;
    end

    aSub = aSub(1:numel(keepEpoch), :);
    EEG_sel = pop_select(EEG_tgt, 'trial', keepEpoch');

    X = double(EEG_sel.data);      % chan x time x trial, uV
    tSec = double(EEG_sel.times) ./ 1000;
    chLabels = chan_labels_string(EEG_sel.chanlocs);

    idxPost = find(ismember(chLabels, ROI_POST));
    idxL = find(ismember(chLabels, ROI_L_POST));
    idxR = find(ismember(chLabels, ROI_R_POST));
    if isempty(idxPost)
        fprintf('[WARN] No posterior ROI channels found for %s\n', subj);
        continue;
    end

    maskP1 = tSec >= WIN_P1(1) & tSec <= WIN_P1(2);
    maskN1 = tSec >= WIN_N1(1) & tSec <= WIN_N1(2);
    nTr = size(X,3);

    p1_amp = nan(nTr,1);
    n1_amp = nan(nTr,1);
    n1_lat_ms = nan(nTr,1);
    p1_lh = nan(nTr,1);
    p1_rh = nan(nTr,1);
    n1_lh = nan(nTr,1);
    n1_rh = nan(nTr,1);
    p1_asym_diff = nan(nTr,1);  % RH-LH
    n1_asym_diff = nan(nTr,1);  % RH-LH
    p1_asym_norm = nan(nTr,1);  % (RH-LH)/(abs(RH)+abs(LH))
    n1_asym_norm = nan(nTr,1);  % (RH-LH)/(abs(RH)+abs(LH))

    for k = 1:nTr
        sigPost = squeeze(mean(X(idxPost,:,k),1));
        p1_amp(k) = mean(sigPost(maskP1), 'omitnan');
        n1_amp(k) = mean(sigPost(maskN1), 'omitnan');

        % N1 latency = latency of most negative point in N1 window
        sigN1 = sigPost(maskN1);
        tN1 = tSec(maskN1);
        if ~isempty(sigN1) && any(isfinite(sigN1))
            [~,ix] = min(sigN1);
            n1_lat_ms(k) = tN1(ix) * 1000;
        end

        if ~isempty(idxL) && ~isempty(idxR)
            sigL = squeeze(mean(X(idxL,:,k),1));
            sigR = squeeze(mean(X(idxR,:,k),1));
            p1_lh(k) = mean(sigL(maskP1), 'omitnan');
            p1_rh(k) = mean(sigR(maskP1), 'omitnan');
            n1_lh(k) = mean(sigL(maskN1), 'omitnan');
            n1_rh(k) = mean(sigR(maskN1), 'omitnan');
            p1_asym_diff(k) = p1_rh(k) - p1_lh(k);
            n1_asym_diff(k) = n1_rh(k) - n1_lh(k);
            p1_asym_norm(k) = p1_asym_diff(k) / (abs(p1_rh(k)) + abs(p1_lh(k)) + eps);
            n1_asym_norm(k) = n1_asym_diff(k) / (abs(n1_rh(k)) + abs(n1_lh(k)) + eps);
        end
    end

    % Build per-trial table
    block_id = aSub.block;
    trial_id = aSub.trial;
    phase = strings(nTr,1);
    correctness = nan(nTr,1);
    rule_level = strings(nTr,1);
    is_acquired = false(nTr,1);
    for k = 1:nTr
        keyk = sprintf('%d_%d', block_id(k), trial_id(k));
        if isKey(phaseMap, keyk); phase(k) = string(phaseMap(keyk)); else; phase(k) = "unknown"; end
        if isKey(corrMap, keyk); correctness(k) = corrMap(keyk); end
        if isKey(rlMap, keyk); rule_level(k) = lower(string(rlMap(keyk))); else; rule_level(k) = "unknown"; end
        is_acquired(k) = any(phase(k) == ["stable","late-found"]);
    end

    T = table;
    T.subj = repmat(subj, nTr, 1);
    T.block_id = block_id;
    T.trial_id = trial_id;
    T.phase = phase;
    T.correctness = correctness;
    T.rule_level = rule_level;
    T.is_acquired = is_acquired;
    T.is_acquired_correct = is_acquired & (correctness == 1);
    T.target_locked_p1_amp_post = p1_amp;
    T.target_locked_n1_amp_post = n1_amp;
    T.target_locked_n1_latency_ms_post = n1_lat_ms;
    T.target_locked_p1_lh = p1_lh;
    T.target_locked_p1_rh = p1_rh;
    T.target_locked_n1_lh = n1_lh;
    T.target_locked_n1_rh = n1_rh;
    T.target_locked_p1_asym_diff = p1_asym_diff;
    T.target_locked_n1_asym_diff = n1_asym_diff;
    T.target_locked_p1_asym_norm = p1_asym_norm;
    T.target_locked_n1_asym_norm = n1_asym_norm;
    T.target_event_num = keepEpoch;
    T.eeg_idx_0based = aSub.eeg_idx;

    nTrialCells = nTrialCells + 1;
    trialRowsCell{nTrialCells} = T;
end

if nTrialCells == 0
    error('No rows generated. Check event code / alignment / metadata columns.');
end

trialRows = vertcat(trialRowsCell{1:nTrialCells});
writetable(trialRows, fullfile(OUT_DIR, 'target_locked_trial_n1_metrics.csv'));

%% Subject-level contrasts in acquired + correct trials
S = trialRows(trialRows.is_acquired_correct == 1 & ismember(trialRows.rule_level, ["global","local"]), :);
subs = unique(S.subj);
rows = table();

for i = 1:numel(subs)
    subj = subs(i);
    D = S(S.subj == subj, :);
    G = D(D.rule_level == "global", :);
    L = D(D.rule_level == "local", :);
    if isempty(G) || isempty(L); continue; end

    r = table;
    r.subj = subj;
    r.n_global_trials = height(G);
    r.n_local_trials = height(L);

    % M1: N1 amplitude contrast (global-local)
    r.n1_amp_global = mean(G.target_locked_n1_amp_post, 'omitnan');
    r.n1_amp_local = mean(L.target_locked_n1_amp_post, 'omitnan');
    r.n1_amp_global_minus_local = r.n1_amp_global - r.n1_amp_local;

    % M2: N1 latency contrast (global-local)
    r.n1_lat_global_ms = mean(G.target_locked_n1_latency_ms_post, 'omitnan');
    r.n1_lat_local_ms = mean(L.target_locked_n1_latency_ms_post, 'omitnan');
    r.n1_lat_global_minus_local_ms = r.n1_lat_global_ms - r.n1_lat_local_ms;

    % M3: asymmetry contrasts (P1/N1, diff + norm)
    r.p1_asym_diff_global = mean(G.target_locked_p1_asym_diff, 'omitnan');
    r.p1_asym_diff_local = mean(L.target_locked_p1_asym_diff, 'omitnan');
    r.p1_asym_diff_global_minus_local = r.p1_asym_diff_global - r.p1_asym_diff_local;

    r.n1_asym_diff_global = mean(G.target_locked_n1_asym_diff, 'omitnan');
    r.n1_asym_diff_local = mean(L.target_locked_n1_asym_diff, 'omitnan');
    r.n1_asym_diff_global_minus_local = r.n1_asym_diff_global - r.n1_asym_diff_local;

    r.p1_asym_norm_global = mean(G.target_locked_p1_asym_norm, 'omitnan');
    r.p1_asym_norm_local = mean(L.target_locked_p1_asym_norm, 'omitnan');
    r.p1_asym_norm_global_minus_local = r.p1_asym_norm_global - r.p1_asym_norm_local;

    r.n1_asym_norm_global = mean(G.target_locked_n1_asym_norm, 'omitnan');
    r.n1_asym_norm_local = mean(L.target_locked_n1_asym_norm, 'omitnan');
    r.n1_asym_norm_global_minus_local = r.n1_asym_norm_global - r.n1_asym_norm_local;

    rows = [rows; r]; %#ok<AGROW>
end

writetable(rows, fullfile(OUT_DIR, 'target_locked_subject_contrasts.csv'));

fprintf('\nDone. Outputs saved in:\n%s\n', OUT_DIR);
fprintf(' - target_locked_trial_n1_metrics.csv\n');
fprintf(' - target_locked_subject_contrasts.csv\n');

%% -------- Local functions --------
function s = event_type_to_string(ev)
    n = numel(ev);
    s = strings(n,1);
    for i = 1:n
        x = ev(i).type;
        if isnumeric(x)
            s(i) = string(num2str(x));
        elseif isstring(x)
            s(i) = x;
        elseif ischar(x)
            s(i) = string(x);
        else
            s(i) = "";
        end
        s(i) = strtrim(s(i));
    end
end

function labels = chan_labels_string(chanlocs)
    n = numel(chanlocs);
    labels = strings(1,n);
    for i = 1:n
        labels(i) = string(strtrim(chanlocs(i).labels));
    end
end

function key = mk_key(block, trial)
    key = strings(numel(block),1);
    for i = 1:numel(block)
        key(i) = sprintf('%d_%d', block(i), trial(i));
    end
end

function c = pick_col(vars, candidates)
    c = "";
    vs = lower(string(vars));
    for i = 1:numel(candidates)
        m = find(vs == lower(candidates(i)), 1);
        if ~isempty(m)
            c = string(vars{m});
            return;
        end
    end
end
