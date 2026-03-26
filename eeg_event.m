%% Extract behavioral trial indices from EEG.event
%
% For every feedback onset (type '60'), reads ruleblockid and trialidx (and fallback
% field names) in EEG chronological order. Writes a long table plus a short per-subject
% summary. Use this sequence as the EEG-native behavior timeline for QC or downstream
% alignment (e.g. match to trials/*.csv on block and trial).
%
% Requires EEGLAB (pop_loadset).
%
% Outputs (under trials_trialwise/<COMP_TAG>/):
%   eeg_event_behavior_sequence.csv
%   eeg_event_behavior_sequence_summary.csv

clear; clc;

BASE = 'Your_BASE';
EEG_DIR = 'Your_EEG_DIR';
COMP_TAG = "1s_comp";

SKIP_SUBS = ["reknow011", "reknow020", "reknow023"];  
FEEDBACK_CODE = "60";

outDir = fullfile(BASE, 'trials_trialwise', COMP_TAG);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
OUT_LONG = fullfile(outDir, 'eeg_event_behavior_sequence.csv');
OUT_SUMMARY = fullfile(outDir, 'eeg_event_behavior_sequence_summary.csv');

setFiles = dir(fullfile(EEG_DIR, 'reknow*_reknow_wcst.set'));
allSubs = strings(numel(setFiles), 1);
for i = 1:numel(setFiles)
    nm = setFiles(i).name;
    allSubs(i) = extractBefore(string(nm), "_reknow_wcst.set");
end
allSubs = unique(allSubs);

allRows = table();
summaryRows = struct('subj', {}, 'n_feedback', {}, 'n_labeled', {}, 'n_block_ge_13', {}, 'note', {});

for s = 1:numel(allSubs)
    subj = allSubs(s);
    if any(subj == SKIP_SUBS)
        continue;
    end

    setName = char(subj + "_reknow_wcst.set");
    fp = fullfile(EEG_DIR, setName);
    if ~isfile(fp)
        fprintf('[SKIP] %s: missing .set\n', subj);
        continue;
    end

    EEG = pop_loadset('filename', setName, 'filepath', EEG_DIR);
    evType = local_event_type_to_string(EEG.event);
    fbIdx = find(evType == FEEDBACK_CODE);
    nFb = numel(fbIdx);

    blk = nan(nFb, 1);
    tr = nan(nFb, 1);
    lat = nan(nFb, 1);
    for k = 1:nFb
        ev = EEG.event(fbIdx(k));
        blk(k) = local_get_num_field(ev, {'ruleblockid', 'block_id', 'block', 'ruleblock'});
        tr(k) = local_get_num_field(ev, {'trialidx', 'trial', 'trial_id'});
        if isfield(ev, 'latency') && isnumeric(ev.latency) && isscalar(ev.latency)
            lat(k) = double(ev.latency);
        end
    end

    eegIdx = (0:nFb-1)';
    T = table(repmat(subj, nFb, 1), eegIdx, blk, tr, lat, ...
        'VariableNames', {'subj', 'eeg_idx', 'block', 'trial', 'latency_samples'});
    allRows = [allRows; T]; %#ok<AGROW>

    nLab = sum(isfinite(blk) & isfinite(tr));
    nB13 = sum(isfinite(blk) & blk >= 13);
    note = "";
    if nLab < nFb
        note = sprintf('%d/%d feedbacks missing block or trial label; ', nFb - nLab, nFb);
    end
    note = char(strtrim(note));

    fprintf('%s: n_feedback=%d n_labeled=%d n_block_ge_13_events=%d\n', subj, nFb, nLab, nB13);

    summaryRows(end+1).subj = char(subj);
    summaryRows(end).n_feedback = nFb;
    summaryRows(end).n_labeled = nLab;
    summaryRows(end).n_block_ge_13 = nB13;
    summaryRows(end).note = note;
end

if ~isempty(allRows)
    writetable(allRows, OUT_LONG);
    fprintf('\nWrote %s (%d rows)\n', OUT_LONG, height(allRows));
end
if ~isempty(summaryRows)
    writetable(struct2table(summaryRows), OUT_SUMMARY);
    fprintf('Wrote %s\n', OUT_SUMMARY);
end

%% Helpers: standardize the data to make sure everything runs smoothly

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
