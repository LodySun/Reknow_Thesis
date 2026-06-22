%% Build eeg_behavior_trial_match_status.csv from EEGLAB events (no SQLite).
%
% For each feedback event (type '60'), read ruleblockid + trialidx and match to
% behavioral trials/trials_trialwise source CSV rows (block>=13). Writes the same
% columns as align_behavior_sqlite.py so build_eeg_tables_feedback_locked.m keeps
% working unchanged (uses eeg_idx as 0-based index among feedback events).
%
% Set WRITE_STANDARD_NAME = true to overwrite trials_trialwise/<COMP_TAG>/eeg_behavior_trial_match_status.csv
% Set false to write ..._from_events.csv for side-by-side comparison first.

clear; clc;

BASE = 'base_dir';
EEG_DIR = 'eeg_dir';
TRIALS_DIR = fullfile(BASE, 'trials');
COMP_TAG = "1s_comp";

WRITE_STANDARD_NAME = true;
SKIP_SUBS = ["reknow011", "reknow020", "reknow023"];
FEEDBACK_CODE = "60";

outDir = fullfile(BASE, 'trials_trialwise', COMP_TAG);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
if WRITE_STANDARD_NAME
    outCsv = fullfile(outDir, 'eeg_behavior_trial_match_status.csv');
else
    outCsv = fullfile(outDir, 'eeg_behavior_trial_match_status_from_events.csv');
end
outSummary = fullfile(outDir, 'eeg_behavior_alignment_from_events_summary.csv');

setFiles = dir(fullfile(EEG_DIR, 'reknow*_reknow_wcst.set'));
allSubs = strings(numel(setFiles), 1);
for i = 1:numel(setFiles)
    nm = setFiles(i).name;
    allSubs(i) = extractBefore(string(nm), "_reknow_wcst.set");
end
allSubs = unique(allSubs);

allRows = table();
sumRows = struct('subj', {}, 'n_beh', {}, 'n_fb', {}, 'n_matched', {}, ...
    'n_missing_in_eeg', {}, 'n_orphan_fb', {}, 'note', {});

for s = 1:numel(allSubs)
    subj = allSubs(s);
    if any(subj == SKIP_SUBS)
        continue;
    end

    behPath = fullfile(TRIALS_DIR, char(subj) + "_trials.csv");
    if ~exist(behPath, 'file')
        fprintf('[SKIP] %s: no %s\n', subj, behPath);
        continue;
    end
    behT = readtable(behPath, 'TextType', 'string');
    if ~ismember('block', behT.Properties.VariableNames) || ~ismember('trial', behT.Properties.VariableNames)
        fprintf('[SKIP] %s: trials.csv missing block/trial\n', subj);
        continue;
    end
    behT.block_num = str2double(string(behT.block));
    behT.trial_num = str2double(string(behT.trial));
    behT = behT(behT.block_num >= 13 & isfinite(behT.block_num) & isfinite(behT.trial_num), :);
    behT = sortrows(behT, {'block_num', 'trial_num'});

    if isempty(behT)
        fprintf('[SKIP] %s: no block>=13 rows\n', subj);
        continue;
    end

    setName = char(subj + "_reknow_wcst.set");
    if ~isfile(fullfile(EEG_DIR, setName))
        fprintf('[SKIP] %s: missing .set\n', subj);
        continue;
    end

    EEG = pop_loadset('filename', setName, 'filepath', EEG_DIR);
    evType = local_event_type_to_string(EEG.event);
    fbIdx = find(evType == FEEDBACK_CODE);
    nFb = numel(fbIdx);

    blk = nan(nFb, 1);
    tr = nan(nFb, 1);
    for k = 1:nFb
        ev = EEG.event(fbIdx(k));
        blk(k) = local_get_num_field(ev, {'ruleblockid', 'block_id', 'block', 'ruleblock'});
        tr(k) = local_get_num_field(ev, {'trialidx', 'trial', 'trial_id'});
    end

    used = false(nFb, 1);
    miss = 0;

    subBlock = behT.block_num;
    subTrial = behT.trial_num;
    nB = numel(subBlock);

    eegIdxCol = nan(nB, 1);
    status = strings(nB, 1);
    tgtTs = nan(nB, 1);
    if ismember('tgt_ts', behT.Properties.VariableNames)
        tgtTs = str2double(string(behT.tgt_ts));
    end

    for i = 1:nB
        b = subBlock(i); t = subTrial(i);
        cand = find(~used & blk == b & tr == t, 1, 'first');
        if isempty(cand)
            status(i) = "missing_in_eeg";
            miss = miss + 1;
        else
            used(cand) = true;
            eegIdxCol(i) = cand - 1;
            status(i) = "matched";
        end
    end

    orphan = sum(~used);
    note = "";
    if any(isnan(blk)) || any(isnan(tr))
        note = char(note + " some_feedback_events_missing_block_or_trialidx;");
    end

    fprintf('%s: beh=%d fb=%d matched=%d missing_in_eeg=%d orphan_fb=%d\n', ...
        subj, nB, nFb, nB - miss, miss, orphan);

    T = table();
    T.subj = repmat(subj, nB, 1);
    T.block = subBlock;
    T.trial = subTrial;
    T.trial_idx_0based = (0:nB-1)';
    T.tgt_ts = tgtTs;
    T.match_status = status;
    T.eeg_idx = eegIdxCol;
    T.eeg_ts_str = repmat("", nB, 1);
    T.time_diff_sec = nan(nB, 1);

    allRows = [allRows; T]; %#ok<AGROW>

    sumRows(end+1).subj = char(subj);
    sumRows(end).n_beh = nB;
    sumRows(end).n_fb = nFb;
    sumRows(end).n_matched = nB - miss;
    sumRows(end).n_missing_in_eeg = miss;
    sumRows(end).n_orphan_fb = orphan;
    sumRows(end).note = note;
end

if ~isempty(allRows)
    writetable(allRows, outCsv);
    fprintf('\nWrote %s (%d rows)\n', outCsv, height(allRows));
end
if ~isempty(sumRows)
    writetable(struct2table(sumRows), outSummary);
    fprintf('Wrote %s\n', outSummary);
end

%% --- helpers ---

function s = local_event_type_to_string(ev)
    n = numel(ev);
    s = strings(n, 1);
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

function v = local_get_num_field(ev, names)
    v = NaN;
    for i = 1:numel(names)
        fn = names{i};
        if isfield(ev, fn)
            x = ev.(fn);
            if isnumeric(x) && isscalar(x)
                v = double(x);
                return;
            end
            if isstring(x) || ischar(x)
                v = str2double(string(x));
                return;
            end
        end
    end
end
