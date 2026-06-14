% MOMO Task3 (QED + DRD2 + Similarity) runner.
% Run this with optimizer1_momo_task3.py using OBJECTIVE_MODE=momo_task3.
%
% This file mirrors no_gui_task1_momo.m:
%   - per-worker communication directory through comm_dir_override
%   - 0-based mol_id written for MOMO warm-start alignment
%   - 250 generations with sparse save: generation 1 plus 10,20,...,250
%   - PlatEMO save = 26

platemo_source_dir = fileparts(mfilename('fullpath'));

if exist('comm_dir_override', 'var') && ~isempty(comm_dir_override)
    matlab_location = comm_dir_override;
    if ~isfolder(matlab_location); mkdir(matlab_location); end
else
    matlab_location = platemo_source_dir;
end

cd(platemo_source_dir);
addpath(genpath(platemo_source_dir));

removeAllFiles(matlab_location);
disp("Deleted communication text files to start a fresh MOMO Task3 run.");
py_shutdown_file = fullfile(matlab_location, 'py_SHUTDOWN.txt');

if ~exist('lead_start', 'var') || ~exist('lead_end', 'var')
    disp('Using default MOMO Task3 lead_range: 1 to all available leads');
    lead_start = 1;
    lead_end = [];
else
    disp(['Using parallel assigned MOMO Task3 lead_range: ', num2str(lead_start), ' to ', num2str(lead_end)]);
end

alg_handle = @FRCSO;
base_seed  = 123456789;

start_run(alg_handle, lead_start, lead_end, base_seed, matlab_location, platemo_source_dir);

writematrix([], py_shutdown_file);

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
        'py_RUN_READY.txt', 'py_RUN_READY_tmp.txt', ...
        'py_EARLY_SUCCESS.txt', 'py_EARLY_SUCCESS_tmp.txt' ...
    };
    for k = 1:numel(names)
        removeFileIfExists(fullfile(matlab_location, names{k}));
    end
end

function start_run(alg, lead_start, lead_end, base_seed, matlab_location, source_dir)
    py_new_run_tmp_file = fullfile(matlab_location, 'py_NEW_RUN_tmp.txt');
    py_new_run_file = fullfile(matlab_location, 'py_NEW_RUN.txt');
    py_lead_tmp_file = fullfile(matlab_location, 'py_LEAD_SMILES_tmp.txt');
    py_lead_file = fullfile(matlab_location, 'py_LEAD_SMILES.txt');
    py_lead_id_tmp_file = fullfile(matlab_location, 'py_LEAD_ID_tmp.txt');
    py_lead_id_file = fullfile(matlab_location, 'py_LEAD_ID.txt');
    py_run_ready_file = fullfile(matlab_location, 'py_RUN_READY.txt');

    lead_smiles = loadTask3LeadSmiles(source_dir);
    if isempty(lead_smiles)
        error('MOMO Task3 lead source is empty.');
    end

    if isempty(lead_end)
        lead_end = numel(lead_smiles);
    end
    run_range = lead_start:lead_end;
    fprintf('Loaded %d MOMO Task3 leads (will run %d to %d)\n', numel(lead_smiles), lead_start, lead_end);

    removeAllFiles(matlab_location);
    for i = run_range
        seed_i = base_seed + i;
        rng(seed_i);

        if i > numel(lead_smiles)
            error('Requested lead index %d exceeds available Task3 leads (%d)', i, numel(lead_smiles));
        end

        lead_idx = i;
        smi = char(lead_smiles(lead_idx));
        fid = fopen(py_lead_tmp_file, 'w');
        if fid < 0
            error('Failed to open lead tmp file: %s', py_lead_tmp_file);
        end
        fprintf(fid, '%s', smi);
        fclose(fid);
        movefile(py_lead_tmp_file, py_lead_file, 'f');

        lead_id0 = lead_idx - 1;
        writematrix([lead_id0], py_lead_id_tmp_file);
        movefile(py_lead_id_tmp_file, py_lead_id_file, 'f');

        if isfile(py_run_ready_file)
            delete(py_run_ready_file);
        end

        writematrix([seed_i], py_new_run_tmp_file);
        movefile(py_new_run_tmp_file, py_new_run_file, 'f');

        fprintf('[MATLAB-MOMO-TASK3] lead_idx=%d lead_id=%d seed=%d smiles=%s\n', lead_idx, lead_id0, seed_i, smi);

        t0 = tic;
        while ~isfile(py_run_ready_file)
            pause(0.01);
            if toc(t0) > 300
                error('Timeout waiting for Python py_RUN_READY.txt');
            end
        end
        delete(py_run_ready_file);

        global platemo_sparse_save_start_generation platemo_sparse_save_interval platemo_sparse_save_keep_first;
        platemo_sparse_save_keep_first = true;
        platemo_sparse_save_start_generation = 10;
        platemo_sparse_save_interval = 10;
        global momo_current_lead_idx;
        momo_current_lead_idx = i;
        platemo('algorithm', alg, 'problem', @DDProblem1, 'maxFE', 25000, 'save', 26);

        % Record mol_id -> .mat file mapping for this lead
        alg_name = func2str(alg);
        mat_filename = sprintf('%s_DDProblem1_M2_D_%d.mat', alg_name, i);
        mapping_file = fullfile(matlab_location, 'mol_id_to_mat_mapping.csv');
        write_header = ~isfile(mapping_file);
        fid_map = fopen(mapping_file, 'a');
        if fid_map >= 0
            if write_header
                fprintf(fid_map, 'mol_id,lead_idx,mat_file,lead_smiles\n');
            end
            lead_smi_safe = strrep(char(smi), ',', ';');
            fprintf(fid_map, '%d,%d,%s,%s\n', lead_id0, i, mat_filename, lead_smi_safe);
            fclose(fid_map);
        end
    end
    clear global platemo_sparse_save_start_generation platemo_sparse_save_interval platemo_sparse_save_keep_first;
end

function lead_smiles = loadTask3LeadSmiles(source_dir)
    candidate_files = {};
    env_lead_file = getenv('TASK3_LEAD_FILE');
    env_data_dir = getenv('TASK3_DATA_DIR');
    if ~isempty(env_lead_file)
        candidate_files{end+1} = env_lead_file;
    end
    if ~isempty(env_data_dir)
        candidate_files{end+1} = fullfile(env_data_dir, 'qeddrd_test.csv');
        candidate_files{end+1} = fullfile(env_data_dir, 'lead_smiles.txt');
    end
    candidate_files{end+1} = fullfile(source_dir, '..', '..', 'MOMO-master-main', 'momo', 'data', 'qeddrd_test.csv');

    for c = 1:numel(candidate_files)
        cand = candidate_files{c};
        if ~isempty(cand) && isfile(cand)
            lines = readlines(cand);
            lines = lines(strlength(strtrim(lines)) > 0);
            lead_smiles = strings(0, 1);
            for k = 1:numel(lines)
                line = strtrim(lines(k));
                if contains(line, ',')
                    first = strtrim(extractBefore(line, ','));
                else
                    parts = split(line);
                    first = strtrim(parts(1));
                end
                first = erase(first, char(65279));
                if strcmpi(first, 'SMILES')
                    continue;
                end
                lead_smiles(end+1, 1) = first; %#ok<AGROW>
            end
            fprintf('MOMO Task3 lead source: %s\n', cand);
            return;
        end
    end

    error('Missing MOMO Task3 lead source. Set TASK3_LEAD_FILE or provide MOMO-master-main/momo/data/qeddrd_test.csv.');
end
