function [T, summary] = compute_momo_task2_metrics_from_platemo(varargin)
% Compute MOMO Task2 metrics (SR, Improvement) from PlatEMO saved runs.
%
% Task2 (pLogP optimization):
%   - Objectives: minimize [-pLogP, -sim]
%   - Success (SR): exists a molecule with pLogP improvement > 0 AND sim >= 0.4
%   - Improvement: best_pLogP - lead_pLogP (positive = improvement)
%
% HV Calculation (MOMO-aligned):
%   - Normalize objectives to [0, 1] range
%   - pLogP normalized using dataset range [-12, 12] (covers typical values)
%   - sim already in [0, 1]
%   - Reference point = [1, 1] (worst normalized point)
%   - This makes HV comparable to MOMO paper (HV in [0, 1])
%
% Usage (in PlatEMO root):
%   [T, summary] = compute_momo_task2_metrics_from_platemo();
%
% Optional name/value:
%   'DataDir'  : directory containing Data/ANSGAIII (default: ./Data/ANSGAIII)
%   'Pattern'  : file glob (default: ANSGAIII_DDProblem1_M2_D*_*.mat)
%   'TreSim'   : similarity threshold (default: 0.4)
%   'LeadPlogpFile' : CSV file with lead pLogP values (default: auto-detect)
%   'PlogpMin' : min pLogP for normalization (default: -12)
%   'PlogpMax' : max pLogP for normalization (default: 12)

    p = inputParser;
    p.addParameter('DataDir', '', @(s)ischar(s) || isstring(s));
    p.addParameter('Pattern', 'ANSGAIII_DDProblem1_M2_D*_*.mat', @(s)ischar(s) || isstring(s));
    p.addParameter('TreSim', 0.4, @(x)isnumeric(x) && isscalar(x));
    p.addParameter('LeadPlogpFile', '', @(s)ischar(s) || isstring(s));
    p.addParameter('PlogpMin', -12, @(x)isnumeric(x) && isscalar(x));
    p.addParameter('PlogpMax', 12, @(x)isnumeric(x) && isscalar(x));
    p.parse(varargin{:});
    opts = p.Results;

    root = fileparts(mfilename('fullpath'));
    cd(root);
    addpath(genpath(root));

    if strlength(string(opts.DataDir)) == 0
        dataDir = fullfile(root, 'Data', 'ANSGAIII');
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
    
    % MOMO-aligned normalization parameters
    plogpMin = opts.PlogpMin;  % -12
    plogpMax = opts.PlogpMax;  % 12
    plogpRange = plogpMax - plogpMin;
    
    % Try to load lead pLogP values
    leadPlogp = nan(N, 1);
    if ~isempty(opts.LeadPlogpFile) && exist(opts.LeadPlogpFile, 'file')
        try
            leadData = readtable(opts.LeadPlogpFile);
            if height(leadData) >= N && any(strcmp(leadData.Properties.VariableNames, 'plogp'))
                leadPlogp = leadData.plogp(1:N);
                fprintf('Loaded lead pLogP values from %s\n', opts.LeadPlogpFile);
            end
        catch
        end
    end

    % Auto-detect MOMO Task2 lead pLogP from dataset if not provided
    if all(isnan(leadPlogp))
        momoPlogpFile = fullfile(root, '..', '..', 'MOMO-master-main', 'momo', 'data', 'logp_test.csv');
        if exist(momoPlogpFile, 'file')
            try
                fid = fopen(momoPlogpFile, 'r');
                leadPlogpAll = [];
                while ~feof(fid)
                    line = fgetl(fid);
                    if ischar(line) && ~isempty(strtrim(line))
                        parts = strsplit(strtrim(line));
                        if numel(parts) >= 2
                            v = str2double(parts{2});
                            leadPlogpAll(end+1, 1) = v; %#ok<AGROW>
                        end
                    end
                end
                fclose(fid);

                for i = 1:N
                    rn = runNums(i);
                    if ~isnan(rn) && rn >= 1 && rn <= numel(leadPlogpAll)
                        leadPlogp(i) = leadPlogpAll(rn);
                    end
                end
                fprintf('Auto-loaded lead pLogP from MOMO dataset: %s\n', momoPlogpFile);
            catch
                try
                    fclose(fid);
                catch
                end
            end
        end
    end

    hvFinal = nan(N, 1);
    hvMax = nan(N, 1);
    successFinal = false(N, 1);
    successAny = false(N, 1);
    bestPlogp = nan(N, 1);
    bestSim = nan(N, 1);
    validFrac = nan(N, 1);
    simEq1Count = nan(N, 1);
    nNonEmptySnaps = nan(N, 1);
    improvement = nan(N, 1);  % best improvement for sim >= 0.4

    treS = opts.TreSim;
    % MOMO-aligned: reference point [1, 1] in normalized space
    ref = [1, 1];

    fprintf('Computing MOMO Task2 metrics from %d runs in %s\n', N, dataDir);
    fprintf('Threshold: SIM>=%.3f | pLogP normalization: [%.1f, %.1f]\n', treS, plogpMin, plogpMax);
    fprintf('HV ref=[%.1f, %.1f] (MOMO-aligned normalized space)\n', ref(1), ref(2));

    for i = 1:N
        path = fullfile(files(i).folder, files(i).name);
        S = load(path, 'result');
        if ~isfield(S, 'result') || isempty(S.result) || size(S.result,2) < 2
            warning('Missing/invalid result in %s', files(i).name);
            continue;
        end

        snapHV = [];
        snapSuccess = false;
        finalObj = [];
        finalObjNorm = [];  % normalized objectives
        finalPlogp = [];
        finalSim = [];
        finalIsValid = [];
        nSnap = 0;

        for r = 1:size(S.result,1)
            if isempty(S.result{r,2})
                continue;
            end
            nSnap = nSnap + 1;
            Pop = S.result{r,2};
            obj = Pop.objs();
            if size(obj,2) ~= 2
                continue;
            end

            % obj(:,1) = -pLogP, obj(:,2) = -sim
            Plogp = -obj(:,1);  % actual pLogP
            Sim = -obj(:,2);    % actual sim
            % Valid: finite, and obj values should be reasonable
            isValid = isfinite(Plogp) & isfinite(Sim) & (obj(:,2) <= 0) & (Plogp < 15);
            
            if any(isValid)
                % Check success: sim >= threshold (improvement checked later with lead plogp)
                snapSuccess = snapSuccess | any(Sim(isValid) >= treS);
                
                % MOMO-aligned HV: normalize objectives to [0, 1]
                % obj1_norm = 1 - (pLogP - plogpMin) / plogpRange  (higher pLogP -> lower obj)
                % obj2_norm = 1 - sim  (higher sim -> lower obj)
                objNorm = zeros(sum(isValid), 2);
                validPlogp = Plogp(isValid);
                validSim = Sim(isValid);
                % Clamp pLogP to normalization range
                validPlogp = max(plogpMin, min(plogpMax, validPlogp));
                objNorm(:,1) = 1 - (validPlogp - plogpMin) / plogpRange;
                objNorm(:,2) = 1 - validSim;
                snapHV(end+1,1) = hv_ref_2d(objNorm, ref);
            end

            finalObj = obj;
            finalPlogp = Plogp;
            finalSim = Sim;
            finalIsValid = isValid;
        end

        if isempty(finalObj)
            continue;
        end

        nNonEmptySnaps(i) = nSnap;
        successAny(i) = snapSuccess;
        if ~isempty(snapHV)
            hvMax(i) = max(snapHV);
        end

        validFrac(i) = mean(finalIsValid);
        if ~any(finalIsValid)
            hvFinal(i) = 0;
            hvMax(i) = max([0; snapHV]);
            successFinal(i) = false;
            continue;
        end

        bestPlogp(i) = max(finalPlogp(finalIsValid));
        bestSim(i) = max(finalSim(finalIsValid));
        simEq1Count(i) = sum(abs(finalSim(finalIsValid) - 1.0) < 1e-9);
        
        % Compute final HV with normalization
        validPlogpFinal = finalPlogp(finalIsValid);
        validSimFinal = finalSim(finalIsValid);
        validPlogpFinal = max(plogpMin, min(plogpMax, validPlogpFinal));
        objNormFinal = zeros(sum(finalIsValid), 2);
        objNormFinal(:,1) = 1 - (validPlogpFinal - plogpMin) / plogpRange;
        objNormFinal(:,2) = 1 - validSimFinal;
        hvFinal(i) = hv_ref_2d(objNormFinal, ref);

        % Compute improvement (only for sim >= threshold)
        maskSim = finalIsValid & (finalSim >= treS);
        if any(maskSim) && ~isnan(leadPlogp(i))
            best_plogp_at_sim = max(finalPlogp(maskSim));
            improvement(i) = best_plogp_at_sim - leadPlogp(i);  % positive = improvement
            successFinal(i) = improvement(i) > 0;
        elseif any(maskSim)
            % No lead plogp, but we have solutions with sim >= threshold
            % Consider it success if pLogP is reasonable (> -5)
            best_plogp_at_sim = max(finalPlogp(maskSim));
            successFinal(i) = best_plogp_at_sim > -5;
            improvement(i) = best_plogp_at_sim;
        end
        % Note: hvFinal already computed above with normalization
    end

    T = table(runNums, string({files.name})', nNonEmptySnaps, ...
        successFinal, successAny, hvFinal, hvMax, ...
        bestPlogp, bestSim, validFrac, simEq1Count, improvement, leadPlogp, ...
        'VariableNames', {'run', 'file', 'nSnaps', ...
        'successFinal', 'successAny', 'hvFinal', 'hvMax', ...
        'bestPlogp', 'bestSim', 'validFrac', 'simEq1Count', 'improvement', 'leadPlogp'});

    summary = struct();
    summary.nRuns = N;
    summary.srCountFinal = sum(successFinal & ~isnan(hvFinal));
    summary.srRateFinal = summary.srCountFinal / N;
    summary.srCountAny = sum(successAny & ~isnan(hvMax));
    summary.srRateAny = summary.srCountAny / N;
    summary.hvFinalMean = mean(hvFinal(~isnan(hvFinal)));
    summary.hvFinalStd = std(hvFinal(~isnan(hvFinal)));
    summary.improvementMean = mean(improvement(~isnan(improvement)));
    summary.improvementStd = std(improvement(~isnan(improvement)));

    fprintf('\nOverall summary (Task2 - pLogP optimization):\n');
    fprintf('  SR (final): %d / %d (%.3f)\n', summary.srCountFinal, N, summary.srRateFinal);
    fprintf('  SR (any)  : %d / %d (%.3f)\n', summary.srCountAny, N, summary.srRateAny);
    fprintf('  HV (final): mean=%.6f, std=%.6f (N=%d)\n', summary.hvFinalMean, summary.hvFinalStd, sum(~isnan(hvFinal)));
    fprintf('  Improvement: mean=%.4f, std=%.4f (N=%d)\n', summary.improvementMean, summary.improvementStd, sum(~isnan(improvement)));

    fprintf('\nPer-run (sorted by run id):\n');
    fprintf('%-6s %-5s %-5s %-7s %-12s %-12s %-10s %-10s %-10s %-10s %-10s\n', ...
        'run', 'fin', 'any', 'snaps', 'hvFinal', 'hvMax', 'bestPlogp', 'bestSim', 'validFrac', 'improve', 'leadPlogp');
    for i = 1:height(T)
        fprintf('%-6d %-5d %-5d %-7d %-12.6f %-12.6f %-10.4f %-10.4f %-10.3f %-10.4f %-10.4f\n', ...
            T.run(i), int32(T.successFinal(i)), int32(T.successAny(i)), int32(T.nSnaps(i)), ...
            T.hvFinal(i), T.hvMax(i), T.bestPlogp(i), T.bestSim(i), T.validFrac(i), ...
            T.improvement(i), T.leadPlogp(i));
    end
end

function hv = hv_ref_2d(points, ref)
% Exact dominated HV in 2D for minimization problems.
    if isempty(points)
        hv = 0;
        return;
    end
    mask = all(points <= ref, 2) & all(isfinite(points), 2);
    points = points(mask, :);
    if isempty(points)
        hv = 0;
        return;
    end
    points = unique(points, 'rows');
    points = sortrows(points, 1);
    nd = zeros(size(points,1), 1, 'logical');
    bestF2 = inf;
    for i = 1:size(points,1)
        if points(i,2) < bestF2
            nd(i) = true;
            bestF2 = points(i,2);
        end
    end
    P = points(nd, :);
    if isempty(P)
        hv = 0;
        return;
    end
    hv = 0;
    prevF2 = ref(2);
    for i = 1:size(P,1)
        width = ref(1) - P(i,1);
        height = prevF2 - P(i,2);
        if width > 0 && height > 0
            hv = hv + width * height;
        end
        prevF2 = P(i,2);
    end
end
