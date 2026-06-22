%% Build EEG tables (feedback-locked) for WCST
% Output files:
%   1) eeg_trial_long.csv
%   2) eeg_block_long.csv
%   3) eeg_subject_traits.csv
%
% Notice:
% - EEG data dir: preprocessed .set/.fdt under 4_ADJUST_IC_CORR
% - Skip subjects: 011, 020, 023
% - Remove missing trials based on alignment outputs (keep abnormal as QC flag)
% - Feedback-locked event code: "60"
% - Alignment CSV: prefer align_behavior_from_eeg_events.m (EEG.event fields);
%   legacy option is Python align_behavior_sqlite.py (SQLite timestamps).

clear; clc;

%% Paths
BASE = '/Users/lodysun/Desktop/Thesis';
EEG_DIR = '/Users/lodysun/Desktop/Thesis/prc/ctap/base_wcst/correct_wcst_5/tppd/4_ADJUST_IC_CORR';
COMP_TAG = "1s_comp";
ALIGN_MATCH_CSV = fullfile(BASE, 'trials_trialwise', COMP_TAG, 'eeg_behavior_trial_match_status.csv');
TRIALWISE_DIR = fullfile(BASE, 'trials_trialwise', 'trialwise');
BLOCK_LABEL_CSV = fullfile(BASE, 'trials_trialwise', 'hmm_mixture', 'hmm_mixture_soft_block_labels.csv');

OUT_DIR = fullfile(BASE, 'trials_trialwise', COMP_TAG, 'eeg_tables');
if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

%% Params
SKIP_SUBS = ["reknow011", "reknow020", "reknow023"];
FEEDBACK_CODE = "60";

EPOCH_WIN_SEC = [-0.2, 0.8];
BASELINE_MS = [-200, 0];

% ERP windows (seconds) - primary windows only
WIN_P3A = [0.30, 0.45];  % frontal / fronto-central (Frontal P3: 300-450 ms)
WIN_P3B = [0.30, 0.50];  % centro-parietal / parietal
WIN_FRN = [0.20, 0.35];  % fronto-central FRN window (feedback-locked)

% Frequency bands (Hz)
BAND_THETA = [4, 7];
BAND_ALPHA = [8, 12];

% Time windows (seconds) - primary windows only
WIN_THETA_MAIN = [0.20, 0.60];   % frontal theta: control/update-related main window
WIN_ALPHA_MAIN = [0.30, 0.80];   % posterior alpha: attentional selection main window

% ROIs
ROI_FRONT = ["Fz","FC1","FC2","Cz","F3","F4"];
ROI_FRN = ["Fz","FC1","FC2","Cz"];
ROI_PARIETAL = ["Pz","P3","P4"];
ROI_ALPHA_POST = ["Pz","P3","P4","Oz","O1","O2","PO9","PO10"];
ROI_VISUAL_POST = ["PO9","PO10","Oz"];
ROI_L = ["F3","F7"];
ROI_R = ["F4","F8"];

%% Read Metadata
alignT = readtable(ALIGN_MATCH_CSV, 'TextType', 'string');

blockTypeT = readtable(BLOCK_LABEL_CSV, 'TextType', 'string');
blockTypeT.block_id = str2double(string(blockTypeT.block_id));

%% Subject List
setFiles = dir(fullfile(EEG_DIR, 'reknow*_reknow_wcst.set'));
allSubs = strings(numel(setFiles),1);
for i = 1:numel(setFiles)
    nm = setFiles(i).name;
    allSubs(i) = extractBefore(string(nm), "_reknow_wcst.set");
end
allSubs = unique(allSubs);

%% Output Holders
trialRowsCell = cell(numel(allSubs), 1);
nTrialCells = 0;

for s = 1:numel(allSubs)
    subj = allSubs(s);
    if any(subj == SKIP_SUBS)
        continue;
    end

    fprintf('\n=== Processing %s ===\n', subj);

    % Behavior trialwise Phase Map
    trialwiseCsv = fullfile(TRIALWISE_DIR, subj + "_trialwise.csv");
    if ~exist(trialwiseCsv, 'file')
        fprintf('[WARN] Missing trialwise file: %s\n', trialwiseCsv);
        continue;
    end
    behT = readtable(trialwiseCsv, 'TextType', 'string');
    behT.block = str2double(string(behT.block_id));
    behT.trial = str2double(string(behT.trial_id));
    behT = behT(:, {'block','trial','phase'});
    behKey = mk_key(behT.block, behT.trial);
    behPhaseMap = containers.Map(cellstr(behKey), cellstr(string(behT.phase)));

    % Block Type Map
    btSub = blockTypeT(blockTypeT.subj == subj, {'block_id','strategy_hard'});
    btMap = containers.Map('KeyType','double', 'ValueType','char');
    for i = 1:height(btSub)
        b = btSub.block_id(i);
        if ~isnan(b)
            btMap(b) = char(btSub.strategy_hard(i));
        end
    end

    % Alignment Rows (Matched Only)
    aSub = alignT(alignT.subj == subj & alignT.match_status == "matched", :);
    if isempty(aSub)
        fprintf('[WARN] No matched alignment rows for %s\n', subj);
        continue;
    end
    aSub.block = str2double(string(aSub.block));
    aSub.trial = str2double(string(aSub.trial));
    aSub.eeg_idx = str2double(string(aSub.eeg_idx));
    aSub.trial_idx_0based = str2double(string(aSub.trial_idx_0based));
    aSub = aSub(~isnan(aSub.eeg_idx), :);

    if isempty(aSub)
        fprintf('[WARN] No matched rows left for %s\n', subj);
        continue;
    end

    % Load EEG
    setName = subj + "_reknow_wcst.set";
    EEG = pop_loadset('filename', char(setName), 'filepath', EEG_DIR);

    % All feedback event indices in original timeline
    evType = event_type_to_string(EEG.event);
    fbEventIdx = find(evType == FEEDBACK_CODE);
    nFb = numel(fbEventIdx);
    if nFb == 0
        fprintf('[WARN] No feedback code %s in %s\n', FEEDBACK_CODE, subj);
        continue;
    end

    % Map aligned rows to feedback event number (1-based)
    aSub.feedback_event_num = aSub.eeg_idx + 1;
    aSub = aSub(aSub.feedback_event_num >= 1 & aSub.feedback_event_num <= nFb, :);
    if isempty(aSub)
        fprintf('[WARN] No valid mapped feedback events after clipping for %s\n', subj);
        continue;
    end

    % Sort by feedback event order
    [~, ord] = sort(aSub.feedback_event_num, 'ascend');
    aSub = aSub(ord, :);
    [~, ia] = unique(aSub.feedback_event_num, 'stable');
    aSub = aSub(ia, :);

    % Epoch all feedback events first, then subset by event number
    EEG_fb = pop_epoch(EEG, {char(FEEDBACK_CODE)}, EPOCH_WIN_SEC, 'epochinfo', 'yes');
    EEG_fb = pop_rmbase(EEG_fb, BASELINE_MS);

    keepEpoch = aSub.feedback_event_num;
    keepEpoch = keepEpoch(keepEpoch <= EEG_fb.trials);
    if isempty(keepEpoch)
        fprintf('[WARN] No epochs left for %s\n', subj);
        continue;
    end

    % Keep same rows
    aSub = aSub(1:numel(keepEpoch), :);
    EEG_sel = pop_select(EEG_fb, 'trial', keepEpoch');

    % data dims: chan x time x trial, in uV
    X = double(EEG_sel.data);
    tSec = double(EEG_sel.times) ./ 1000;
    srate = double(EEG_sel.srate);
    chLabels = chan_labels_string(EEG_sel.chanlocs);

    % indices
    idxFront = find(ismember(chLabels, ROI_FRONT));
    idxFRN = find(ismember(chLabels, ROI_FRN));
    idxPar = find(ismember(chLabels, ROI_PARIETAL));
    idxAlpha = find(ismember(chLabels, ROI_ALPHA_POST));
    idxVisualPost = find(ismember(chLabels, ROI_VISUAL_POST));
    idxL = find(ismember(chLabels, ROI_L));
    idxR = find(ismember(chLabels, ROI_R));

    maskP3a = tSec >= WIN_P3A(1) & tSec <= WIN_P3A(2);
    maskP3b = tSec >= WIN_P3B(1) & tSec <= WIN_P3B(2);
    maskFRN = tSec >= WIN_FRN(1) & tSec <= WIN_FRN(2);

    nTr = size(X,3);
    p3a = nan(nTr,1);
    p3b = nan(nTr,1);
    frn = nan(nTr,1);
    thetaPow = nan(nTr,1);
    alphaPow = nan(nTr,1);
    visualAlphaPow = nan(nTr,1);
    asymAlpha = nan(nTr,1);

    for k = 1:nTr
        if ~isempty(idxFront)
            sigFront = squeeze(mean(X(idxFront,:,k), 1)); % 1 x time
            p3a(k) = mean(sigFront(maskP3a), 'omitnan');
            thetaPow(k) = band_power_window(sigFront, tSec, srate, BAND_THETA, WIN_THETA_MAIN);
        end
        if ~isempty(idxFRN)
            sigFRN = squeeze(mean(X(idxFRN,:,k), 1)); % feedback-locked fronto-central
            frn(k) = mean(sigFRN(maskFRN), 'omitnan'); % more negative => stronger FRN
        end
        if ~isempty(idxPar)
            sigPar = squeeze(mean(X(idxPar,:,k), 1));
            p3b(k) = mean(sigPar(maskP3b), 'omitnan');
        end
        if ~isempty(idxAlpha)
            sigAlpha = squeeze(mean(X(idxAlpha,:,k), 1));
            alphaPow(k) = band_power_window(sigAlpha, tSec, srate, BAND_ALPHA, WIN_ALPHA_MAIN);
        end
        if ~isempty(idxVisualPost)
            sigVisualPost = squeeze(mean(X(idxVisualPost,:,k), 1));
            visualAlphaPow(k) = band_power_window(sigVisualPost, tSec, srate, BAND_ALPHA, WIN_ALPHA_MAIN);
        end
        if ~isempty(idxL) && ~isempty(idxR)
            sigL = squeeze(mean(X(idxL,:,k), 1));
            sigR = squeeze(mean(X(idxR,:,k), 1));
            % L/R log-power asym (F3,F7 vs F4,F8), alpha band, WIN_ALPHA_MAIN
            pL = band_power_window(sigL, tSec, srate, BAND_ALPHA, WIN_ALPHA_MAIN);
            pR = band_power_window(sigR, tSec, srate, BAND_ALPHA, WIN_ALPHA_MAIN);
            if isfinite(pL) && isfinite(pR) && pL > 0 && pR > 0
                asymAlpha(k) = log(pL) - log(pR);
            end
        end
    end

    % build per-trial metadata
    block_id = aSub.block;
    trial_id = aSub.trial;
    phase = strings(nTr,1);
    block_type = strings(nTr,1);
    for k = 1:nTr
        key = sprintf('%d_%d', block_id(k), trial_id(k));
        if isKey(behPhaseMap, key)
            phase(k) = string(behPhaseMap(key));
        else
            phase(k) = "unknown";
        end
        if isKey(btMap, block_id(k))
            block_type(k) = string(btMap(block_id(k)));
        else
            block_type(k) = "unknown";
        end
    end

    T = table;
    T.subj = repmat(subj, nTr, 1);
    T.block_id = block_id;
    T.trial_id = trial_id;
    T.phase = phase;
    T.block_type = block_type;
    T.feedback_locked_P3a = p3a;
    T.feedback_locked_P3b = p3b;
    T.feedback_locked_FRN = frn;
    T.feedback_locked_theta_power = thetaPow;
    T.feedback_locked_alpha_power = alphaPow;
    T.feedback_locked_visual_alpha_power = visualAlphaPow;
    T.theta_power = thetaPow; % backward-compatible alias
    T.alpha_power = alphaPow; % backward-compatible alias
    T.visual_alpha_power = visualAlphaPow; % backward-compatible alias
    T.frontal_asym_alpha = asymAlpha; % backward-compatible alias
    T.feedback_locked_frontal_asym_alpha = asymAlpha;
    T.qc_kept_after_alignment = true(nTr,1);
    T.feedback_event_num = keepEpoch;
    T.eeg_idx_0based = aSub.eeg_idx;

    nTrialCells = nTrialCells + 1;
    trialRowsCell{nTrialCells} = T;
end

if nTrialCells == 0
    error('No trial rows generated. Check paths / events / alignment files.');
end

trialRows = vertcat(trialRowsCell{1:nTrialCells});

%% Save eeg_trial_long
eeg_trial_long = trialRows;
writetable(eeg_trial_long, fullfile(OUT_DIR, 'eeg_trial_long.csv'));

%% Build eeg_block_long
% Aggregate by subj x block x phase x block_type
[G, subjG, blockG, phaseG, typeG] = findgroups( ...
    eeg_trial_long.subj, eeg_trial_long.block_id, eeg_trial_long.phase, eeg_trial_long.block_type);

n_trial = splitapply(@numel, eeg_trial_long.trial_id, G);
mP3a = splitapply(@(x) mean(x,'omitnan'), eeg_trial_long.feedback_locked_P3a, G);
mP3b = splitapply(@(x) mean(x,'omitnan'), eeg_trial_long.feedback_locked_P3b, G);
mFRN = splitapply(@(x) mean(x,'omitnan'), eeg_trial_long.feedback_locked_FRN, G);
mTheta = splitapply(@(x) mean(x,'omitnan'), eeg_trial_long.feedback_locked_theta_power, G);
mAlpha = splitapply(@(x) mean(x,'omitnan'), eeg_trial_long.feedback_locked_alpha_power, G);
mVisualAlpha = splitapply(@(x) mean(x,'omitnan'), eeg_trial_long.feedback_locked_visual_alpha_power, G);
mAsym = splitapply(@(x) mean(x,'omitnan'), eeg_trial_long.feedback_locked_frontal_asym_alpha, G);

eeg_block_long = table;
eeg_block_long.subj = subjG;
eeg_block_long.block_id = blockG;
eeg_block_long.phase = phaseG;
eeg_block_long.block_type = typeG;
eeg_block_long.n_trials = n_trial;
eeg_block_long.feedback_locked_P3a = mP3a;
eeg_block_long.feedback_locked_P3b = mP3b;
eeg_block_long.feedback_locked_FRN = mFRN;
eeg_block_long.feedback_locked_theta_power = mTheta;
eeg_block_long.feedback_locked_alpha_power = mAlpha;
eeg_block_long.feedback_locked_visual_alpha_power = mVisualAlpha;
eeg_block_long.theta_power = mTheta; % backward-compatible alias
eeg_block_long.alpha_power = mAlpha; % backward-compatible alias
eeg_block_long.visual_alpha_power = mVisualAlpha; % backward-compatible alias
eeg_block_long.frontal_asym_alpha = mAsym; % backward-compatible alias
eeg_block_long.feedback_locked_frontal_asym_alpha = mAsym;

writetable(eeg_block_long, fullfile(OUT_DIR, 'eeg_block_long.csv'));

%% Build eeg_subject_traits
subs = unique(eeg_block_long.subj);
eeg_subject_traits = table();

for i = 1:numel(subs)
    subj = subs(i);
    B = eeg_block_long(eeg_block_long.subj == subj, :);

    Gd = B(B.block_type == "gradual_like", :);
    Go = B(B.block_type == "one_shot_like", :);

    row = table;
    row.subj = subj;

    row.mean_P3b_gradual = mean(Gd.feedback_locked_P3b, 'omitnan');
    row.mean_P3b_one_shot = mean(Go.feedback_locked_P3b, 'omitnan');
    row.delta_P3b_gradual_minus_one_shot = row.mean_P3b_gradual - row.mean_P3b_one_shot;

    row.mean_FRN_gradual = mean(Gd.feedback_locked_FRN, 'omitnan');
    row.mean_FRN_one_shot = mean(Go.feedback_locked_FRN, 'omitnan');
    row.delta_FRN_gradual_minus_one_shot = row.mean_FRN_gradual - row.mean_FRN_one_shot;

    row.mean_theta_gradual = mean(Gd.feedback_locked_theta_power, 'omitnan');
    row.mean_theta_one_shot = mean(Go.feedback_locked_theta_power, 'omitnan');
    row.delta_theta_gradual_minus_one_shot = row.mean_theta_gradual - row.mean_theta_one_shot;

    row.mean_visual_alpha_gradual = mean(Gd.feedback_locked_visual_alpha_power, 'omitnan');
    row.mean_visual_alpha_one_shot = mean(Go.feedback_locked_visual_alpha_power, 'omitnan');
    row.delta_visual_alpha_gradual_minus_one_shot = row.mean_visual_alpha_gradual - row.mean_visual_alpha_one_shot;

    row.mean_asym_gradual = mean(Gd.feedback_locked_frontal_asym_alpha, 'omitnan');
    row.mean_asym_one_shot = mean(Go.feedback_locked_frontal_asym_alpha, 'omitnan');
    row.delta_asym_gradual_minus_one_shot = row.mean_asym_gradual - row.mean_asym_one_shot;

    row.n_blocks_gradual = height(Gd);
    row.n_blocks_one_shot = height(Go);
    row.n_trials_total = sum(B.n_trials, 'omitnan');

    eeg_subject_traits = [eeg_subject_traits; row]; %#ok<AGROW>
end

writetable(eeg_subject_traits, fullfile(OUT_DIR, 'eeg_subject_traits.csv'));

fprintf('\nDone. Outputs saved in:\n%s\n', OUT_DIR);
fprintf(' - eeg_trial_long.csv\n');
fprintf(' - eeg_block_long.csv\n');
fprintf(' - eeg_subject_traits.csv\n');

%% Local Functions
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

function p = band_power_1d(sig, srate, bandHz)
    sig = double(sig(:));
    sig = sig - mean(sig, 'omitnan');
    if any(isnan(sig)) || numel(sig) < 16
        p = NaN;
        return;
    end
    % Use a simple one-sided FFT-based PSD estimate to avoid
    % Signal Processing Toolbox dependency (no pwelch needed).
    nfft = max(256, 2^nextpow2(numel(sig)));
    x = fft(sig, nfft);
    p2 = (abs(x).^2) / (srate * nfft);     % two-sided PSD
    p1 = p2(1:floor(nfft/2)+1);            % one-sided PSD
    if numel(p1) > 2
        p1(2:end-1) = 2 * p1(2:end-1);
    end
    f = (0:floor(nfft/2))' * (srate / nfft);

    m = (f >= bandHz(1)) & (f <= bandHz(2));
    if ~any(m)
        p = NaN;
        return;
    end
    fm = f(m);
    pm = p1(m);
    if numel(pm) == 1
        % Single-bin case: trapz would return 0; use PSD * bin width.
        if numel(f) > 1
            df = f(2) - f(1);
        else
            df = 1;
        end
        p = pm(1) * df;
    else
        p = trapz(fm, pm);
    end
end

function p = band_power_window(sig, tSec, srate, bandHz, winSec)
    sig = double(sig(:));
    tSec = double(tSec(:));
    m = (tSec >= winSec(1)) & (tSec <= winSec(2));
    if ~any(m)
        p = NaN;
        return;
    end
    p = band_power_1d(sig(m), srate, bandHz);
end
