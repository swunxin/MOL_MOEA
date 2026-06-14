% Headless BindingDB P2 runner.
% P2 objective is implemented in optimizer_task13.py:
%   OBJECTIVE_MODE=bindingdb_p2
%   QED + DRD2 + Similarity
%   success: QED >= 0.80, DRD2 >= 0.40, sim >= 0.30
%
% Start optimizer_task13.py first, then run this MATLAB script.

platemo_source_dir = fileparts(mfilename('fullpath'));

if exist('comm_dir_override', 'var') && ~isempty(comm_dir_override)
    matlab_location = char(comm_dir_override);
else
    env_comm = getenv('PLATEMO_COMM_DIR');
    if ~isempty(env_comm)
        matlab_location = env_comm;
    else
        matlab_location = platemo_source_dir;
    end
end
if ~isfolder(matlab_location)
    mkdir(matlab_location);
end

cd(platemo_source_dir);
addpath(genpath(platemo_source_dir));

fprintf('[MATLAB-P2] source_dir=%s\n', platemo_source_dir);
fprintf('[MATLAB-P2] comm_dir=%s\n', matlab_location);

removeAllFiles(matlab_location);
disp('[MATLAB-P2] Deleted communication text files to start a fresh run.');
py_shutdown_file = fullfile(matlab_location, 'py_SHUTDOWN.txt');

if ~exist('lead_start', 'var') || ~exist('lead_end', 'var')
    disp('[MATLAB-P2] Using default lead_range: 1 to all available leads');
    lead_start = 1;
    lead_end = [];
else
    fprintf('[MATLAB-P2] Using assigned lead_range: %d to %d\n', lead_start, lead_end);
end

alg_name = getenv('PLATEMO_ALGORITHM');
if isempty(alg_name)
    alg_name = 'ANSGAIII';
end
alg_handle = str2func(alg_name);

base_seed = readEnvDouble('PLATEMO_BASE_SEED', 123456789);
max_fe = round(readEnvDouble('PLATEMO_MAXFE', 25000));
if max_fe <= 0
    error('PLATEMO_MAXFE must be positive.');
end

env_auto_retry = firstNonEmptyEnv({'TASK2_AUTO_RETRY', 'TASK1_AUTO_RETRY'});
auto_retry = any(strcmpi(strtrim(env_auto_retry), {'1','true','yes','on'}));
env_max_attempts = firstNonEmptyEnv({'TASK2_MAX_ATTEMPTS', 'TASK1_MAX_ATTEMPTS'});
if isempty(env_max_attempts)
    max_attempts = 1;
else
    max_attempts = str2double(env_max_attempts);
    if isnan(max_attempts) || max_attempts < 1
        error('Invalid max-attempts environment value: %s', env_max_attempts);
    end
end
max_attempts = round(max_attempts);
if ~auto_retry
    max_attempts = 1;
end

tre_q = readEnvDouble('TASK2_SUCCESS_QED', 0.80);
tre_d = readEnvDouble('TASK2_SUCCESS_DRD2', 0.40);
tre_s = readEnvDouble('TASK2_SUCCESS_SIM', 0.30);

fprintf('[MATLAB-P2] algorithm=%s base_seed=%d maxFE=%d autoRetry=%d maxAttempts=%d thresholds=(QED>=%.3f,DRD2>=%.3f,SIM>=%.3f)\n', ...
    alg_name, base_seed, max_fe, auto_retry, max_attempts, tre_q, tre_d, tre_s);

start_run(alg_handle, alg_name, lead_start, lead_end, base_seed, max_fe, auto_retry, max_attempts, tre_q, tre_d, tre_s, matlab_location, platemo_source_dir);

writematrix([], py_shutdown_file);
fprintf('[MATLAB-P2] Wrote shutdown file: %s\n', py_shutdown_file);

function removeFileIfExists(path)
    if isfile(path)
        delete(path);
    end
end

function removeAllFiles(matlab_location)
    names = {
        'py_SHUTDOWN.txt', ...
        'py_OBJ.txt', 'py_OBJ_tmp.txt', ...
        'py_EMB.txt', 'py_EMB_tmp.txt', ...
        'py_LOWER.txt', 'py_LOWER_tmp.txt', ...
        'py_UPPER.txt', 'py_UPPER_tmp.txt', ...
        'py_M.txt', 'py_M_tmp.txt', ...
        'py_N.txt', 'py_N_tmp.txt', ...
        'py_init_pop.txt', 'py_init_pop_tmp.txt', ...
        'matlab_REPAIR_EMB.txt', 'matlab_REPAIR_EMB_tmp.txt', ...
        'py_NEW_RUN.txt', 'py_NEW_RUN_tmp.txt', ...
        'py_LEAD_SMILES.txt', 'py_LEAD_SMILES_tmp.txt', ...
        'py_LEAD_ID.txt', 'py_LEAD_ID_tmp.txt', ...
        'py_RUN_READY.txt', 'py_RUN_READY_tmp.txt' ...
    };
    for k = 1:numel(names)
        removeFileIfExists(fullfile(matlab_location, names{k}));
    end
end

function value = readEnvDouble(name, default_value)
    raw = getenv(name);
    if isempty(raw)
        value = default_value;
        return;
    end
    value = str2double(raw);
    if isnan(value)
        error('Invalid numeric environment variable %s: %s', name, raw);
    end
end

function value = firstNonEmptyEnv(names)
    value = '';
    for k = 1:numel(names)
        raw = getenv(names{k});
        if ~isempty(raw)
            value = raw;
            return;
        end
    end
end

function start_run(alg, alg_name, lead_start, lead_end, base_seed, max_fe, auto_retry, max_attempts, tre_q, tre_d, tre_s, matlab_location, source_dir)
    py_new_run_tmp_file = fullfile(matlab_location, 'py_NEW_RUN_tmp.txt');
    py_new_run_file = fullfile(matlab_location, 'py_NEW_RUN.txt');
    py_lead_tmp_file = fullfile(matlab_location, 'py_LEAD_SMILES_tmp.txt');
    py_lead_file = fullfile(matlab_location, 'py_LEAD_SMILES.txt');
    py_lead_id_tmp_file = fullfile(matlab_location, 'py_LEAD_ID_tmp.txt');
    py_lead_id_file = fullfile(matlab_location, 'py_LEAD_ID.txt');
    py_run_ready_file = fullfile(matlab_location, 'py_RUN_READY.txt');

    lead_smiles = loadLeadSmiles(source_dir, 'TASK2_LEAD_FILE', 'TASK2_DATA_DIR', 'p2_lead_smiles.txt', 'p2_drd2_qs_lead_smiles.txt');
    lead_smiles = cleanLeadLines(lead_smiles);
    if isempty(lead_smiles)
        error('P2 lead source is empty.');
    end

    run_range = getRunRange('TASK2_MOL_IDS', lead_start, lead_end, numel(lead_smiles));
    fprintf('[MATLAB-P2] Loaded %d leads (will run %s)\n', numel(lead_smiles), rangeToText(run_range));

    removeAllFiles(matlab_location);

    for i = run_range
        if i > numel(lead_smiles)
            error('Requested lead index %d exceeds available leads (%d)', i, numel(lead_smiles));
        end

        lead_idx = i;
        lead_id0 = lead_idx - 1;
        smi = char(lead_smiles(lead_idx));
        lead_success = false;

        for attempt = 1:max_attempts
            seed_i = base_seed + lead_idx + (attempt - 1) * 100000000;
            rng(seed_i);

            fid = fopen(py_lead_tmp_file, 'w');
            if fid < 0
                error('Failed to open lead tmp file: %s', py_lead_tmp_file);
            end
            fprintf(fid, '%s', smi);
            fclose(fid);
            movefile(py_lead_tmp_file, py_lead_file, 'f');

            writematrix([lead_id0], py_lead_id_tmp_file);
            movefile(py_lead_id_tmp_file, py_lead_id_file, 'f');

            if isfile(py_run_ready_file)
                delete(py_run_ready_file);
            end

            writematrix([seed_i], py_new_run_tmp_file);
            movefile(py_new_run_tmp_file, py_new_run_file, 'f');

            fprintf('[MATLAB-P2-RUN] lead_idx=%d lead_id=%d attempt=%d/%d seed=%d\n', lead_idx, lead_id0, attempt, max_attempts, seed_i);

            waitForRunReady(py_run_ready_file, matlab_location);

            delete(py_run_ready_file);
            global momo_current_lead_idx;
            momo_current_lead_idx = lead_idx;

            before_result_files = snapshot_result_files(source_dir, alg_name, 3);
            platemo('algorithm', alg, 'problem', @DDProblem1, 'maxFE', max_fe, 'save', 100000);

            result_file = find_new_result_file(source_dir, alg_name, 3, before_result_files);
            [success_final, success_any, best_q, best_d, best_s] = p2_success_from_result(result_file, tre_q, tre_d, tre_s);
            append_retry_summary(matlab_location, lead_id0, lead_idx, attempt, seed_i, success_final, success_any, best_q, best_d, best_s, result_file);
            fprintf('[MATLAB-P2-RETRY] lead_id=%d attempt=%d/%d successFinal=%d successAny=%d bestQED=%.4f bestDRD2=%.4f bestSim=%.4f\n', ...
                lead_id0, attempt, max_attempts, success_final, success_any, best_q, best_d, best_s);

            if success_any
                lead_success = true;
                break;
            end

            if auto_retry && attempt < max_attempts
                fprintf('[MATLAB-P2-RETRY] lead_id=%d failed attempt %d; retrying with a new seed.\n', lead_id0, attempt);
            end
        end

        if auto_retry && ~lead_success
            fprintf('[MATLAB-P2-RETRY] lead_id=%d failed after %d attempts.\n', lead_id0, max_attempts);
        end
    end
end

function lead_smiles = loadLeadSmiles(source_dir, file_env, dir_env, default_name, fallback_name)
    candidate_files = {};
    env_lead_file = getenv(file_env);
    env_data_dir = getenv(dir_env);
    if ~isempty(env_lead_file)
        candidate_files{end + 1} = env_lead_file;
    end
    if ~isempty(env_data_dir)
        candidate_files{end + 1} = fullfile(env_data_dir, default_name);
        candidate_files{end + 1} = fullfile(env_data_dir, fallback_name);
        candidate_files{end + 1} = fullfile(env_data_dir, 'lead_smiles.txt');
    end
    candidate_files{end + 1} = fullfile(source_dir, '..', 'data', 'leads', default_name);
    candidate_files{end + 1} = fullfile(source_dir, '..', 'data', 'leads', fallback_name);

    for c = 1:numel(candidate_files)
        cand = candidate_files{c};
        if ~isempty(cand) && isfile(cand)
            lead_smiles = readlines(cand);
            fprintf('[MATLAB-P2] Lead source: %s\n', cand);
            return;
        end
    end
    error('Missing P2 lead source. Set %s or provide data/leads/%s.', file_env, default_name);
end

function lead_smiles = cleanLeadLines(lead_smiles)
    lead_smiles = lead_smiles(strlength(lead_smiles) > 0);
    for k = 1:numel(lead_smiles)
        line = lead_smiles(k);
        if contains(line, ',')
            lead_smiles(k) = extractBefore(line, ',');
        end
    end
end

function run_range = getRunRange(mol_ids_env, lead_start, lead_end, n_leads)
    env_mol_ids = getenv(mol_ids_env);
    if ~isempty(env_mol_ids)
        mol_ids = parseIdList(env_mol_ids);
        if isempty(mol_ids)
            error('%s was set but no valid numeric mol_id was parsed: %s', mol_ids_env, env_mol_ids);
        end
        run_range = mol_ids + 1;
        return;
    end
    if isempty(lead_end)
        lead_end = n_leads;
    end
    run_range = lead_start:lead_end;
end

function ids = parseIdList(raw)
    parts = regexp(char(raw), '[,;\s]+', 'split');
    ids = [];
    for k = 1:numel(parts)
        token = strtrim(parts{k});
        if isempty(token)
            continue;
        end
        val = str2double(token);
        if isnan(val) || val < 0 || abs(val - round(val)) > 1e-9
            error('Invalid mol_id: %s', token);
        end
        ids(end + 1) = round(val); %#ok<AGROW>
    end
    ids = unique(ids, 'stable');
end

function waitForRunReady(py_run_ready_file, matlab_location)
    t0 = tic;
    while ~isfile(py_run_ready_file)
        pause(0.05);
        if toc(t0) > 300
            error('Timeout waiting for Python py_RUN_READY.txt in %s', matlab_location);
        end
    end
end

function text = rangeToText(run_range)
    if isempty(run_range)
        text = '<empty>';
    elseif numel(run_range) <= 8
        text = char(strjoin(string(run_range), ','));
    else
        text = sprintf('%d:%d (%d leads)', run_range(1), run_range(end), numel(run_range));
    end
end

function names = snapshot_result_files(source_dir, alg_name, objective_dim)
    data_dir = fullfile(source_dir, 'Data', alg_name);
    pattern = sprintf('%s_DDProblem1_M%d_D*_*.mat', alg_name, objective_dim);
    files = dir(fullfile(data_dir, pattern));
    if isempty(files)
        names = strings(0, 1);
    else
        names = string({files.name})';
    end
end

function result_file = find_new_result_file(source_dir, alg_name, objective_dim, before_names)
    data_dir = fullfile(source_dir, 'Data', alg_name);
    pattern = sprintf('%s_DDProblem1_M%d_D*_*.mat', alg_name, objective_dim);
    files = dir(fullfile(data_dir, pattern));
    if isempty(files)
        error('Could not find PlatEMO result file in %s with pattern %s', data_dir, pattern);
    end
    names = string({files.name})';
    is_new = ~ismember(names, before_names);
    if any(is_new)
        files = files(is_new);
    else
        warning('No newly named PlatEMO result file detected; using most recently modified file.');
    end
    [~, order] = sort([files.datenum], 'descend');
    f = files(order(1));
    result_file = fullfile(f.folder, f.name);
end

function [success_final, success_any, best_q, best_d, best_s] = p2_success_from_result(result_file, tre_q, tre_d, tre_s)
    S = load(result_file, 'result');
    if ~isfield(S, 'result') || isempty(S.result) || size(S.result, 2) < 2
        error('Missing/invalid result in %s', result_file);
    end

    success_any = false;
    final_q = [];
    final_d = [];
    final_s = [];
    final_valid = [];
    for r = 1:size(S.result, 1)
        if isempty(S.result{r, 2})
            continue;
        end
        Pop = S.result{r, 2};
        obj = Pop.objs();
        if size(obj, 2) ~= 3
            continue;
        end
        q = -obj(:, 1);
        d = -obj(:, 2);
        sim = -obj(:, 3);
        valid = all(isfinite(obj), 2) & all(obj <= 0, 2);
        if any(valid)
            success_any = success_any | any((q(valid) >= tre_q) & (d(valid) >= tre_d) & (sim(valid) >= tre_s));
        end
        final_q = q;
        final_d = d;
        final_s = sim;
        final_valid = valid;
    end

    if isempty(final_q) || ~any(final_valid)
        success_final = false;
        best_q = nan;
        best_d = nan;
        best_s = nan;
        return;
    end

    success_final = any((final_q(final_valid) >= tre_q) & (final_d(final_valid) >= tre_d) & (final_s(final_valid) >= tre_s));
    best_q = max(final_q(final_valid));
    best_d = max(final_d(final_valid));
    best_s = max(final_s(final_valid));
end

function append_retry_summary(matlab_location, lead_id0, lead_idx, attempt, seed_i, success_final, success_any, best_q, best_d, best_s, result_file)
    out_dir = fileparts(matlab_location);
    if isempty(out_dir)
        out_dir = matlab_location;
    end
    path = fullfile(out_dir, 'retry_summary.csv');
    write_header = ~isfile(path);
    fid = fopen(path, 'a');
    if fid < 0
        warning('Could not append retry summary: %s', path);
        return;
    end
    cleaner = onCleanup(@()fclose(fid));
    if write_header
        fprintf(fid, 'mol_id,lead_idx,attempt,seed,success_final,success_any,best_qed,best_drd2,best_sim,result_file\n');
    end
    fprintf(fid, '%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,"%s"\n', ...
        lead_id0, lead_idx, attempt, seed_i, success_final, success_any, best_q, best_d, best_s, result_file);
    clear cleaner;
end
