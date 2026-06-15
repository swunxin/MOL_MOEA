% RUN THIS FIRST, THEN THE PYTHON SCRIPT!

% Source dir: always this script's own directory (for MATLAB/PlatEMO source files)
platemo_source_dir = fileparts(mfilename('fullpath'));

% Communication dir: supports parallel execution.
% Set comm_dir_override before running to use a separate per-worker directory.
% Example: matlab -r "comm_dir_override='/tmp/w0'; lead_start=1; lead_end=200; run('no_gui_task1.m'); exit"
if exist('comm_dir_override', 'var') && ~isempty(comm_dir_override)
    matlab_location = comm_dir_override;
    if ~isfolder(matlab_location); mkdir(matlab_location); end
else
    matlab_location = platemo_source_dir;
end

% Make sure PlatEMO (including Problems/SOLUTION.m) is on the MATLAB path
cd(platemo_source_dir);
addpath(genpath(platemo_source_dir));

removeAllFiles(matlab_location);
disp("Deleted communication text files to start a fresh run.");
py_shutdown_file = fullfile(matlab_location, '', 'py_SHUTDOWN.txt');

% 支持并行：判断 sh 脚本是否通过命令行参数传入了区间起止点
if ~exist('lead_start', 'var') || ~exist('lead_end', 'var')
    % 如果没有传入，则使用默认区间（适合本地单进程运行）
    lead_start = 1;
    lead_end   = 800;
    disp(['Using default override lead_range: ', num2str(lead_start), ' to ', num2str(lead_end)]);
else
    disp(['Using parallel assigned lead_range: ', num2str(lead_start), ' to ', num2str(lead_end)]);
end

alg_handle = getMomoAlgorithmHandle();
base_seed  = 123456789;


start_run(alg_handle, lead_start, lead_end, base_seed, matlab_location, platemo_source_dir);

writematrix([], py_shutdown_file);

function removeFileIfExists(path)
    if isfile(path)
        delete(path)
    end
end
function removeAllFiles(matlab_location)
    py_shutdown_file = fullfile(matlab_location, '', 'py_SHUTDOWN.txt');
    py_new_run_file = fullfile(matlab_location, '', 'py_NEW_RUN.txt');
    removeFileIfExists(py_shutdown_file);
    removeFileIfExists(fullfile(matlab_location, '', 'py_OBJ.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_OBJ_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_EMB.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_EMB_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_LOWER.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_LOWER_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_UPPER.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_UPPER_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_M.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_M_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_N.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_N_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_init_pop.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_init_pop_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'matlab_REPAIR_EMB.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'matlab_REPAIR_EMB_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_NEW_RUN_tmp.txt'));
    removeFileIfExists(py_new_run_file);
    removeFileIfExists(fullfile(matlab_location, '', 'py_LEAD_SMILES.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_LEAD_SMILES_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_LEAD_ID.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_LEAD_ID_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_RUN_READY.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_RUN_READY_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_EARLY_SUCCESS.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_EARLY_SUCCESS_tmp.txt'));
end

function start_run(alg, lead_start, lead_end, base_seed, matlab_location, source_dir)
    if nargin < 6; source_dir = matlab_location; end
    py_new_run_tmp_file = fullfile(matlab_location, '', 'py_NEW_RUN_tmp.txt');
    py_new_run_file = fullfile(matlab_location, '', 'py_NEW_RUN.txt');

    % Lead SMILES list priority:
    % 1) TASK1_LEAD_FILE
    % 2) TASK1_DATA_DIR/lead_smiles.txt
    % 3) ../data/<TASK1_DATASET_NAME>/lead_smiles.txt
    dataset_name = getenv('TASK1_DATASET_NAME');
    if isempty(dataset_name)
        dataset_name = 'task1_zinc_qed06_08_clean100';
    end

    candidate_files = {};
    env_lead_file = getenv('TASK1_LEAD_FILE');
    env_data_dir = getenv('TASK1_DATA_DIR');
    if ~isempty(env_lead_file)
        candidate_files{end+1} = env_lead_file;
    end
    if ~isempty(env_data_dir)
        candidate_files{end+1} = fullfile(env_data_dir, 'lead_smiles.txt');
    end
    candidate_files{end+1} = fullfile(source_dir, '..', 'data', dataset_name, 'lead_smiles.txt');

    lead_source = '';
    for c = 1:numel(candidate_files)
        cand = candidate_files{c};
        if ~isempty(cand) && isfile(cand)
            lead_smiles = readlines(cand);
            lead_source = cand;
            break
        end
    end

    if ~exist('lead_smiles', 'var')
        error('Missing lead source. Set TASK1_LEAD_FILE or TASK1_DATA_DIR, or provide ../data/%s/lead_smiles.txt.', dataset_name);
    end

    lead_smiles = lead_smiles(strlength(lead_smiles) > 0);
    if isempty(lead_smiles)
        error('Lead source is empty.');
    end

    % If lines are CSV, keep only first field
    for k = 1:numel(lead_smiles)
        line = lead_smiles(k);
        if contains(line, ',')
            lead_smiles(k) = extractBefore(line, ',');
        end
    end

    if isempty(lead_end)
        lead_end = numel(lead_smiles);
    end
    run_range = lead_start:lead_end;
    fprintf('Loaded %d Task1 leads from %s (will run %d to %d)\n', numel(lead_smiles), lead_source, lead_start, lead_end);

    py_lead_tmp_file = fullfile(matlab_location, '', 'py_LEAD_SMILES_tmp.txt');
    py_lead_file = fullfile(matlab_location, '', 'py_LEAD_SMILES.txt');

    py_lead_id_tmp_file = fullfile(matlab_location, '', 'py_LEAD_ID_tmp.txt');
    py_lead_id_file = fullfile(matlab_location, '', 'py_LEAD_ID.txt');

    py_run_ready_file = fullfile(matlab_location, '', 'py_RUN_READY.txt');
    py_early_success_file = fullfile(matlab_location, '', 'py_EARLY_SUCCESS.txt');

    removeAllFiles(matlab_location);
    for i = run_range
        seed_i = base_seed + i;
        rng(seed_i); % sets the seed to your desired value

        % Write lead SMILES for this run (1-based line index)
        if i > numel(lead_smiles)
            error('Requested lead index %d exceeds available leads (%d)', i, numel(lead_smiles));
        end
        lead_idx = i;
        fid = fopen(py_lead_tmp_file, 'w'); fprintf(fid, '%s', lead_smiles(lead_idx)); fclose(fid);
        movefile(py_lead_tmp_file, py_lead_file);

        % Write 0-based mol_id for MOMO alignment
        lead_id0 = lead_idx - 1;
        writematrix([lead_id0], py_lead_id_tmp_file);
        movefile(py_lead_id_tmp_file, py_lead_id_file);

        % Clear handshake flags before starting the run
        if isfile(py_run_ready_file)
            delete(py_run_ready_file);
        end
        if isfile(py_early_success_file)
            delete(py_early_success_file);
        end

        writematrix([seed_i], py_new_run_tmp_file);
        movefile(py_new_run_tmp_file, py_new_run_file);

        % Wait for Python to finish setting up this run (and possibly early-stop)
        t0 = tic;
        while ~isfile(py_run_ready_file)
            pause(0.01);
            if toc(t0) > 300
                error('Timeout waiting for Python py_RUN_READY.txt');
            end
        end

        % If Python signaled early success, skip PlatEMO for this lead
        if isfile(py_early_success_file)
            disp('Early success signaled by Python; skipping PlatEMO.');
            delete(py_early_success_file);
            delete(py_run_ready_file);
            continue
        end

        % Consume ready flag before running PlatEMO
        delete(py_run_ready_file);
        % 250代：nPop=100, nIter=250 => maxFE=25000
        % For this MOMO run, keep generation 1 plus 10, 20, ..., 250 in
        % result (26 populations total). See the sparse-save hook in
        % ALGORITHM.m.
        global platemo_sparse_save_start_generation platemo_sparse_save_interval platemo_sparse_save_keep_first;
        platemo_sparse_save_keep_first = true;
        platemo_sparse_save_start_generation = 10;
        platemo_sparse_save_interval = 10;
        global momo_current_lead_idx;
        momo_current_lead_idx = i;
        % --- MOL_MOEA 必须传参数 cell（尤其 task=1）：否则 ParameterSet 全取默认
        %     => task=0 => objThr=[] => POP_BANK 输出合并关闭 => 退化为纯 v7
        %     => 热启的达标分子丢了就丢（这正是 SR 只有 44% 的根因）。
        %     非 MOL_MOEA 算法（FRCSO/PEATSD/...）不吃这些参数，仍用裸句柄。 ---
        alg_arg = alg;
        this_alg_name = func2str(alg);
        if startsWith(this_alg_name, 'MOL_MOEA')
            inj = double(contains(this_alg_name, '_inj'));   % bank=0 / bank_inj=1 (L2 注入)
            % ParameterSet 顺序：D,tau0,H,entropyBonus,learnRate,bankOn,mergeOut,
            %                    injectOn,inject_max,r_frac,K_max,task  （[] = 用该算法自己的默认）
            %   K_max 传 [] -> 各算法用自身默认(v10_bank=30, v11_inj=200), 便于"放大BANK"按版本区分。
            alg_arg = {alg, [],0.7,5,0,0.01, 1,1,inj,5,0.1,[], 1};
        end
        platemo('algorithm',alg_arg,'problem',@DDProblem1,'maxFE',25000,'save',26);

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
            lead_smi_safe = strrep(char(lead_smiles(i)), ',', ';');
            fprintf(fid_map, '%d,%d,%s,%s\n', lead_id0, i, mat_filename, lead_smi_safe);
            fclose(fid_map);
        end
    end
    clear global platemo_sparse_save_start_generation platemo_sparse_save_interval platemo_sparse_save_keep_first;
end

function alg = getMomoAlgorithmHandle()
% Returns algorithm handle. Must be set via PLATEMO_ALGORITHM env var.
    env_alg = getenv('PLATEMO_ALGORITHM');
    if isempty(env_alg)
        error('PLATEMO_ALGORITHM environment variable is not set. Please specify the algorithm name (e.g. FRCSO_N100, PEATSD).');
    end
    alg = str2func(strtrim(env_alg));
    fprintf('[MATLAB-MOMO] Using algorithm: %s\n', env_alg);
end
