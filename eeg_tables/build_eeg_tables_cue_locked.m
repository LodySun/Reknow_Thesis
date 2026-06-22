%% Build cue-locked trial table for within-level vs cross-level shift (EEG strand 3)
% Output: eeg_trial_cue_locked.csv (subj, block_id, trial_id, frontal_theta_cue,
%   posterior_alpha_cue, P3a_cue, shift_type)
%
% Copy of feedback-locked pipeline with:
% - Epoch to CUE event code "10" instead of feedback "60"
% - Same trial order via alignment (cue epoch i = feedback epoch i)
% - Cue-locked time windows: frontal theta 0.2–0.6 s, posterior alpha 0.4–1.0 s, P3a 0.25–0.45 s
% - shift_type from trialwise rule_level (within_level / cross_level per block)
%
% Usage (pick one):
%   1) cd to project root first, then: run('codes/EEG/build_eeg_tables_cue_locked.m')
%   2) Or use full path: run("path_placeholder")

clear; clc;

%% -------------------- Paths --------------------
BASE = 'base_dir';
EEG_DIR = 'eeg_dir';
COMP_TAG = '1s_comp';  % '' 或 '2s' 使用旧目录；'1s_comp' 使用新目录
if strcmp(COMP_TAG, '') || strcmp(COMP_TAG, '2s') || strcmp(COMP_TAG, 'default')
    ALIGN_MATCH_CSV = fullfile(BASE, 'trials_trialwise', 'eeg_behavior_trial_match_status.csv');
    OUT_DIR = fullfile(BASE, 'trials_trialwise', 'eeg_tables');
else
    ALIGN_MATCH_CSV = fullfile(BASE, 'trials_trialwise', COMP_TAG, 'eeg_behavior_trial_match_status.csv');
    OUT_DIR = fullfile(BASE, 'trials_trialwise', COMP_TAG, 'eeg_tables');
end
TRIALWISE_DIR = fullfile(BASE, 'trials_trialwise', 'trialwise');
if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

%% -------------------- Params --------------------
SKIP_SUBS = ["reknow011", "reknow020", "reknow023"];
FEEDBACK_CODE = "60";  % used only to get alignment indices
CUE_CODE = "10";

EPOCH_WIN_SEC = [-0.2, 1.0];   % cue-locked window (longer for alpha)
BASELINE_MS = [-200, 0];

% Cue-locked time windows (seconds)
WIN_THETA_CUE = [0.20, 0.60];  % frontal theta
WIN_ALPHA_CUE = [0.40, 1.00];  % posterior alpha
WIN_P3A_CUE = [0.25, 0.45];    % frontal P3a

BAND_THETA = [4, 7];
BAND_ALPHA = [8, 12];
ROI_FRONT = ["Fz","FC1","FC2","Cz","F3","F4"];
ROI_ALPHA_POST = ["Pz","P3","P4","Oz","O1","O2","PO9","PO10"];

%% -------------------- Read alignment --------------------
alignT = readtable(ALIGN_MATCH_CSV, 'TextType', 'string');

%% -------------------- Subject list --------------------
setFiles = dir(fullfile(EEG_DIR, 'reknow*_reknow_wcst.set'));
allSubs = strings(numel(setFiles),1);
for i = 1:numel(setFiles)
    nm = setFiles(i).name;
    allSubs(i) = extractBefore(string(nm), "_reknow_wcst.set");
end
allSubs = unique(allSubs);

%% -------------------- Output holder --------------------
trialRows = table();

for s = 1:numel(allSubs)
    subj = allSubs(s);
    if any(subj == SKIP_SUBS)
        continue;
    end

    fprintf('\n=== Cue-locked %s ===\n', subj);

    % -------- Trialwise: phase + rule_level for shift_type --------
    trialwiseCsv = fullfile(TRIALWISE_DIR, subj + "_trialwise.csv");
    if ~exist(trialwiseCsv, 'file')
        fprintf('[WARN] Missing trialwise file: %s\n', trialwiseCsv);
        continue;
    end
    behT = readtable(trialwiseCsv, 'TextType', 'string');
    behT.block = str2double(string(behT.block_id));
    [~, ia] = unique(behT.block, 'stable');
    blockRule = behT(ia, {'block'});
    if ismember('rule_level', behT.Properties.VariableNames)
        blockRule.rule_level = behT.rule_level(ia);
    else
        blockRule.rule_level = repmat(string(missing), height(blockRule), 1);
    end
    blockRule = sortrows(blockRule, 'block');
    nB = height(blockRule);
    shift_type = cell(nB, 1);
    for i = 1:nB
        if i == 1 || ismissing(blockRule.rule_level(i)) || ismissing(blockRule.rule_level(max(1,i-1)))
            shift_type{i} = 'missing';
        else
            prev = char(string(blockRule.rule_level(i-1)));
            curr = char(string(blockRule.rule_level(i)));
            if strcmp(strtrim(prev), strtrim(curr))
                shift_type{i} = 'within_level';
            else
                shift_type{i} = 'cross_level';
            end
        end
    end
    blockRule.shift_type = shift_type;

    % -------- Alignment (same as feedback-locked) --------
    aSub = alignT(alignT.subj == subj & alignT.match_status == "matched", :);
    if isempty(aSub)
        fprintf('[WARN] No matched alignment rows for %s\n', subj);
        continue;
    end
    aSub.block = str2double(string(aSub.block));
    aSub.trial = str2double(string(aSub.trial));
    aSub.eeg_idx = str2double(string(aSub.eeg_idx));
    aSub = aSub(~isnan(aSub.eeg_idx), :);
    if isempty(aSub)
        continue;
    end

    % -------- Load EEG --------
    setName = subj + "_reknow_wcst.set";
    EEG = pop_loadset('filename', char(setName), 'filepath', EEG_DIR);

    evType = event_type_to_string(EEG.event);
    fbEventIdx = find(evType == FEEDBACK_CODE);
    nFb = numel(fbEventIdx);
    if nFb == 0
        fprintf('[WARN] No feedback code %s in %s\n', FEEDBACK_CODE, subj);
        continue;
    end

    aSub.feedback_event_num = aSub.eeg_idx + 1;
    aSub = aSub(aSub.feedback_event_num >= 1 & aSub.feedback_event_num <= nFb, :);
    if isempty(aSub)
        continue;
    end
    [~, ord] = sort(aSub.feedback_event_num, 'ascend');
    aSub = aSub(ord, :);
    [~, ia] = unique(aSub.feedback_event_num, 'stable');
    aSub = aSub(ia, :);

    % -------- Epoch to CUE (same trial order: select by feedback_event_num) --------
    EEG_cue = pop_epoch(EEG, {char(CUE_CODE)}, EPOCH_WIN_SEC, 'epochinfo', 'yes');
    EEG_cue = pop_rmbase(EEG_cue, BASELINE_MS);

    keepEpoch = aSub.feedback_event_num;
    keepEpoch = keepEpoch(keepEpoch <= EEG_cue.trials);
    if isempty(keepEpoch)
        fprintf('[WARN] No cue epochs left for %s\n', subj);
        continue;
    end

    aSub = aSub(1:numel(keepEpoch), :);
    EEG_sel = pop_select(EEG_cue, 'trial', keepEpoch');

    X = double(EEG_sel.data);
    tSec = double(EEG_sel.times) ./ 1000;
    srate = double(EEG_sel.srate);
    chLabels = chan_labels_string(EEG_sel.chanlocs);

    idxFront = find(ismember(chLabels, ROI_FRONT));
    idxAlpha = find(ismember(chLabels, ROI_ALPHA_POST));

    maskP3a = tSec >= WIN_P3A_CUE(1) & tSec <= WIN_P3A_CUE(2);

    nTr = size(X, 3);
    frontal_theta_cue = nan(nTr, 1);
    posterior_alpha_cue = nan(nTr, 1);
    P3a_cue = nan(nTr, 1);

    for k = 1:nTr
        if ~isempty(idxFront)
            sigFront = squeeze(mean(X(idxFront,:,k), 1));
            P3a_cue(k) = mean(sigFront(maskP3a), 'omitnan');
            frontal_theta_cue(k) = band_power_window(sigFront, tSec, srate, BAND_THETA, WIN_THETA_CUE);
        end
        if ~isempty(idxAlpha)
            sigAlpha = squeeze(mean(X(idxAlpha,:,k), 1));
            posterior_alpha_cue(k) = band_power_window(sigAlpha, tSec, srate, BAND_ALPHA, WIN_ALPHA_CUE);
        end
    end

    block_id = aSub.block;
    trial_id = aSub.trial;

    % Map block_id -> shift_type
    shift_type_col = repmat(string('missing'), nTr, 1);
    for k = 1:nTr
        bid = block_id(k);
        idx = find(blockRule.block == bid, 1);
        if ~isempty(idx)
            shift_type_col(k) = string(blockRule.shift_type{idx});
        end
    end

    T = table;
    T.subj = repmat(subj, nTr, 1);
    T.block_id = block_id;
    T.trial_id = trial_id;
    T.frontal_theta_cue = frontal_theta_cue;
    T.posterior_alpha_cue = posterior_alpha_cue;
    T.P3a_cue = P3a_cue;
    T.shift_type = shift_type_col;

    trialRows = [trialRows; T]; %#ok<AGROW>
end

if isempty(trialRows)
    error('No cue-locked trial rows. Check paths / events / alignment.');
end

%% -------------------- Save --------------------
writetable(trialRows, fullfile(OUT_DIR, 'eeg_trial_cue_locked.csv'));
fprintf('\nDone. Saved: %s\n', fullfile(OUT_DIR, 'eeg_trial_cue_locked.csv'));
if ismember('shift_type', trialRows.Properties.VariableNames)
    fprintf('shift_type counts:\n');
    disp(tabulate(trialRows.shift_type));
end

%% -------------------- Local (same as feedback-locked) --------------------
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

function p = band_power_1d(sig, srate, bandHz)
    sig = double(sig(:));
    sig = sig - mean(sig, 'omitnan');
    if any(isnan(sig)) || numel(sig) < 16
        p = NaN;
        return;
    end
    nfft = max(256, 2^nextpow2(numel(sig)));
    x = fft(sig, nfft);
    p2 = (abs(x).^2) / (srate * nfft);
    p1 = p2(1:floor(nfft/2)+1);
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
