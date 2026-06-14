function [T, summary] = compute_momo_task3_metrics_from_platemo(varargin)
% Compute MOMO Task3 metrics (SR, HV) from PlatEMO saved runs.
%
% Task3 (QED + DRD2 + Similarity - 3 objectives):
%   - Objectives: minimize [-QED, -DRD2, -sim]
%   - Success (SR): exists a molecule with QED >= 0.8 AND DRD2 >= 0.4 AND sim >= 0.3
%
% HV Calculation:
%   - Normalize objectives to [0, 1] range
%   - QED already in [0, 1]
%   - DRD2 already in [0, 1]
%   - sim already in [0, 1]
%   - Reference point = [1, 1, 1] (worst normalized point)
%
% Usage (in PlatEMO root):
%   [T, summary] = compute_momo_task3_metrics_from_platemo();
%
% Optional name/value:
%   'DataDir'  : directory containing Data/ANSGAIII (default: ./Data/ANSGAIII)
%   'Pattern'  : file glob (default: ANSGAIII_DDProblem1_M3_D*_*.mat)
%   'TreQed'   : QED threshold (default: 0.8)
%   'TreDrd2'  : DRD2 threshold (default: 0.4)
%   'TreSim'   : similarity threshold (default: 0.3)

    p = inputParser;
    p.addParameter('DataDir', '', @(s)ischar(s) || isstring(s));
    p.addParameter('Pattern', 'LMOCSO_DDProblem1_M3_D*_*.mat', @(s)ischar(s) || isstring(s));
    p.addParameter('TreQed', 0.8, @(x)isnumeric(x) && isscalar(x));
    p.addParameter('TreDrd2', 0.4, @(x)isnumeric(x) && isscalar(x));
    p.addParameter('TreSim', 0.3, @(x)isnumeric(x) && isscalar(x));
    p.parse(varargin{:});
    opts = p.Results;

    root = fileparts(mfilename('fullpath'));
    cd(root);
    addpath(genpath(root));

    if strlength(string(opts.DataDir)) == 0
        dataDir = fullfile(root, 'Data', 'LMOCSO');
    else
        dataDir = char(opts.DataDir);
    end
    if exist(dataDir, 'dir') ~= 7
        error('Data directory not found: %s', dataDir);
    end

    files = dir(fullfile(dataDir, char(opts.Pattern)));
    if isempty(files)
        error('No result files found in: %s (pattern: %s)', dataDir, char(opts.Pattern));
    end

    % Sort by trailing run number
    runNums = nan(numel(files), 1);
    for i = 1:numel(files)
        tok = regexp(files(i).name, '_(\d+)\.mat$', 'tokens', 'once');
        if ~isempty(tok)
            runNums(i) = str2double(tok{1});
        end
    end
    [~, order] = sort(runNums);
    files = files(order);
    runNums = runNums(order);

    N = numel(files);
    fprintf('Found %d result files.\n', N);

    % Pre-allocate result arrays
    results = struct('runNum', num2cell(runNums), ...
                     'success', num2cell(zeros(N,1)), ...
                     'bestQed', num2cell(nan(N,1)), ...
                     'bestDrd2', num2cell(nan(N,1)), ...
                     'bestSim', num2cell(nan(N,1)), ...
                     'HV', num2cell(nan(N,1)), ...
                     'paretoSize', num2cell(zeros(N,1)), ...
                     'nSnaps', num2cell(zeros(N,1)), ...
                     'drd2ObjMin', num2cell(nan(N,1)), ...
                     'drd2ObjMax', num2cell(nan(N,1)));

    for i = 1:N
        fname = fullfile(dataDir, files(i).name);
        data = load(fname, 'result');
        if ~isfield(data, 'result') || isempty(data.result) || size(data.result,2) < 2
            fprintf('Run %3d: invalid result format in %s\n', runNums(i), files(i).name);
            continue;
        end

        % Follow task1/task2 style: iterate snapshots in result(:,2),
        % and use the final non-empty snapshot as "final population".
        final_obj = [];
        nSnap = 0;
        for r = 1:size(data.result,1)
            if isempty(data.result{r,2})
                continue;
            end
            nSnap = nSnap + 1;
            Pop = data.result{r,2};
            try
                obj = Pop.objs();
            catch
                % Fallback for older saved formats
                if isstruct(Pop) && isfield(Pop, 'objs')
                    obj = Pop.objs;
                else
                    obj = [];
                end
            end
            if ~isempty(obj) && size(obj,2) >= 3
                final_obj = obj(:,1:3);
            end
        end

        if isempty(final_obj)
            fprintf('Run %3d: no valid objective snapshots in %s\n', runNums(i), files(i).name);
            continue;
        end

        results(i).nSnaps = nSnap;

        % Stored objectives are negative (minimize -QED, -DRD2, -sim)
        % Convert back to original scale
        qed_vals = -final_obj(:, 1);   % QED in [0, 1]
        drd2_vals = -final_obj(:, 2);  % DRD2 in [0, 1]
        sim_vals = -final_obj(:, 3);   % sim in [0, 1]

        % Keep finite rows only (duplicate penalties may push obj > 0)
        valid = isfinite(qed_vals) & isfinite(drd2_vals) & isfinite(sim_vals);
        if ~any(valid)
            fprintf('Run %3d: no finite objective rows in %s\n', runNums(i), files(i).name);
            continue;
        end
        qed_vals = qed_vals(valid);
        drd2_vals = drd2_vals(valid);
        sim_vals = sim_vals(valid);
        obj_valid = final_obj(valid, :);

        % Check success condition
        success_mask = (qed_vals >= opts.TreQed) & ...
                       (drd2_vals >= opts.TreDrd2) & ...
                       (sim_vals >= opts.TreSim);
        results(i).success = any(success_mask);
        
        % Best values
        results(i).bestQed = max(qed_vals);
        results(i).bestDrd2 = max(drd2_vals);
        results(i).bestSim = max(sim_vals);
        results(i).drd2ObjMin = min(obj_valid(:,2));
        results(i).drd2ObjMax = max(obj_valid(:,2));
        
        % Pareto front size
        results(i).paretoSize = size(obj_valid, 1);
        
        % HV calculation (all objectives already in [0,1], ref=[1,1,1])
        % obj in storage is negative, so we compute HV on -obj normalized
        obj_norm = zeros(size(obj_valid));
        obj_norm(:, 1) = 1 + obj_valid(:, 1);  % -QED in [-1,0] -> [0,1]
        obj_norm(:, 2) = 1 + obj_valid(:, 2);  % -DRD2 in [-1,0] -> [0,1]
        obj_norm(:, 3) = 1 + obj_valid(:, 3);  % -sim in [-1,0] -> [0,1]
        
        % Clip to [0, 1] for safety
        obj_norm = max(0, min(1, obj_norm));
        
        % HV with reference point [1, 1, 1]
        refPoint = [1, 1, 1];
        hv = hv_ref_3d(obj_norm, refPoint);
        results(i).HV = hv;
        
        fprintf('Run %3d: success=%d, QED=%.3f, DRD2=%.6f, sim=%.3f, HV=%.4f, snaps=%d, obj2=[%.4f, %.4f]\n', ...
                runNums(i), results(i).success, results(i).bestQed, ...
                results(i).bestDrd2, results(i).bestSim, results(i).HV, ...
                results(i).nSnaps, results(i).drd2ObjMin, results(i).drd2ObjMax);
    end

    % Build output table
    T = struct2table(results);
    
    % Summary statistics
    SR = mean([results.success]) * 100;
    meanHV = mean([results.HV], 'omitnan');
    stdHV = std([results.HV], 'omitnan');
    meanQed = mean([results.bestQed], 'omitnan');
    meanDrd2 = mean([results.bestDrd2], 'omitnan');
    meanSim = mean([results.bestSim], 'omitnan');
    
    summary = struct();
    summary.nRuns = N;
    summary.SR = SR;
    summary.meanHV = meanHV;
    summary.stdHV = stdHV;
    summary.meanBestQed = meanQed;
    summary.meanBestDrd2 = meanDrd2;
    summary.meanBestSim = meanSim;
    
    fprintf('\n========== Task3 Summary ==========\n');
    fprintf('Total runs: %d\n', N);
    fprintf('Success Rate (SR): %.2f%% (QED>=%.1f, DRD2>=%.1f, sim>=%.1f)\n', ...
            SR, opts.TreQed, opts.TreDrd2, opts.TreSim);
    fprintf('Hypervolume: %.4f ± %.4f\n', meanHV, stdHV);
    fprintf('Avg best QED:  %.4f\n', meanQed);
    fprintf('Avg best DRD2: %.4f\n', meanDrd2);
    fprintf('Avg best sim:  %.4f\n', meanSim);
    fprintf('====================================\n');
end

function hv = hv_ref_3d(points, ref)
% Dominated HV in 3D (minimization) with robust fallback.
% points: Nx3 objective matrix, ideally in [0,1], and better = smaller.
% ref   : 1x3 reference point.

    if isempty(points)
        hv = 0;
        return;
    end

    mask = all(isfinite(points), 2) & all(points <= ref, 2);
    P = points(mask, :);
    if isempty(P)
        hv = 0;
        return;
    end
    P = unique(P, 'rows');

    % Prefer exact HV if STK toolbox is available.
    if exist('stk_dominatedhv', 'file') == 2
        try
            hv = stk_dominatedhv(P, ref);
            return;
        catch
        end
    end

    % Fallback: Monte-Carlo estimate in [0,ref] box.
    % Deterministic seed for reproducibility.
    rng(12345, 'twister');
    nSample = 20000;
    box = rand(nSample, 3) .* ref;
    dominated = false(nSample,1);
    for j = 1:size(P,1)
        dominated = dominated | all(box >= P(j,:), 2);
    end
    hv = mean(dominated) * prod(ref);
end
