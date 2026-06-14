function [T, summary] = compute_momo_task1_metrics_from_platemo(varargin)
% Compute MOMO Task1 metrics (SR, HV) from PlatEMO saved runs.
%
% This aligns with MOMO's Task1(QED) definitions found in MOMO_task1.py and
% GB-GA-P_task1_nsga3.py:
%   - Success (SR): exists a molecule with qed >= 0.9 AND sim >= 0.4.
%   - Hypervolume (HV): compute dominated HV on points [-qed, -sim] with
%     reference point [0,0] (pygmo style).
%
% Notes on "final" vs "any" snapshot metrics:
%   PlatEMO saves multiple snapshots in result{:,2}. MOMO implementations
%   often count success if it appears at any generation and may track HV
%   per generation. Therefore, this script reports BOTH:
%     - Final snapshot metrics: successFinal, hvFinal
%     - Any snapshot metrics : successAny (OR over snapshots), hvMax (max HV)
%
% Usage (in PlatEMO root):
%   [T, summary] = compute_momo_task1_metrics_from_platemo();
%
% Optional name/value:
%   'DataDir'  : directory containing .mat result files (default: ./Data/FRCSO_N100)
%   'Pattern'  : file glob (default: FRCSO_N100_DDProblem1_M2_D*_*.mat)
%   'TreQED'   : QED threshold (default: 0.9)
%   'TreSim'   : similarity threshold (default: 0.4)
%   'RefPoint' : 1x2 reference point in objective space (default: [0 0])
%   'OutputDir': directory for CSV/TXT outputs (default: DataDir/metrics)
%   'WriteFiles': write per-run CSV and summary TXT (default: true)
%   'ShowPerRun': true/false/'auto'; auto prints rows only for small N (default: 'auto')
%   'MaxPrintRows': maximum N printed when ShowPerRun='auto' (default: 100)
%   'SaveDecs' : save final-population decision variables for SMILES decoding (default: true)

    p = inputParser;
    p.addParameter('DataDir', '', @(s)ischar(s) || isstring(s));
    p.addParameter('Pattern', 'FRCSO_N100_DDProblem1_M2_D*_*.mat', @(s)ischar(s) || isstring(s));
    p.addParameter('TreQED', 0.9, @(x)isnumeric(x) && isscalar(x));
    p.addParameter('TreSim', 0.4, @(x)isnumeric(x) && isscalar(x));
    p.addParameter('RefPoint', [0 0], @(x)isnumeric(x) && isequal(size(x),[1 2]));
    p.addParameter('OutputDir', '', @(s)ischar(s) || isstring(s));
    p.addParameter('WriteFiles', true, @(x)islogical(x) || isnumeric(x));
    p.addParameter('ShowPerRun', 'auto', @(x)islogical(x) || isnumeric(x) || ischar(x) || isstring(x));
    p.addParameter('MaxPrintRows', 100, @(x)isnumeric(x) && isscalar(x) && x >= 0);
    p.addParameter('SaveDecs', true, @(x)islogical(x) || isnumeric(x));
    p.parse(varargin{:});
    opts = p.Results;

    root = fileparts(mfilename('fullpath'));
    cd(root);
    addpath(genpath(root));

    if strlength(string(opts.DataDir)) == 0
        dataDir = fullfile(root, 'Data', 'FRCSO_N100');
    else
        dataDir = char(opts.DataDir);
    end
    if exist(dataDir, 'dir') ~= 7
        error('Data directory not found: %s', dataDir);
    end
    if strlength(string(opts.OutputDir)) == 0
        outputDir = fullfile(dataDir, 'metrics');
    else
        outputDir = char(opts.OutputDir);
    end
    writeFiles = logical(opts.WriteFiles);
    saveDecs = logical(opts.SaveDecs);
    if writeFiles && exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end

    files = dir(fullfile(dataDir, char(opts.Pattern)));
    if isempty(files)
        error('No result files found in: %s (pattern: %s)', dataDir, char(opts.Pattern));
    end

    % Sort by trailing run number (avoids lexicographic 1,10,11,...,2)
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
    hvFinal = nan(N, 1);
    hvMax = nan(N, 1);
    successFinal = false(N, 1);
    successAny = false(N, 1);
    bestQ = nan(N, 1);
    bestS = nan(N, 1);
    validFrac = nan(N, 1);
    simEq1Count = nan(N, 1);
    nNonEmptySnaps = nan(N, 1);

    % Near-success diagnostics (final snapshot)
    maxQ_at_sim_ge_treS = nan(N, 1);
    maxS_at_q_ge_treQ = nan(N, 1);
    count_sim_ge_treS = nan(N, 1);
    count_q_ge_treQ = nan(N, 1);
    count_both = nan(N, 1);
    leadLikeQ = nan(N, 1);

    treQ = opts.TreQED;
    treS = opts.TreSim;
    ref = opts.RefPoint;

    % Decision-variable collection (for post-hoc SMILES decoding)
    allDecsRows = {};
    allMappingRows = {};

    fprintf('Computing MOMO Task1 metrics from %d runs in %s\n', N, dataDir);
    fprintf('Thresholds: QED>=%.3f, SIM>=%.3f | HV ref=[%.3f, %.3f] (objective space)\n', treQ, treS, ref(1), ref(2));

    for i = 1:N
        path = fullfile(files(i).folder, files(i).name);
        S = load(path, 'result');
        if ~isfield(S, 'result') || isempty(S.result) || size(S.result,2) < 2
            warning('Missing/invalid result in %s', files(i).name);
            continue;
        end

        % Iterate all non-empty snapshots
        snapHV = [];
        snapSuccess = false;
        finalObj = [];
        finalQ = [];
        finalSim = [];
        finalDecs = [];
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

            Q = -obj(:,1);
            Sim = -obj(:,2);
            isValid = isfinite(Q) & isfinite(Sim) & (obj(:,1) <= 0) & (obj(:,2) <= 0);
            if any(isValid)
                snapSuccess = snapSuccess | any((Q(isValid) >= treQ) & (Sim(isValid) >= treS));
                snapHV(end+1,1) = hv_ref_2d(obj(isValid,:), ref); %#ok<AGROW>
            end

            finalDecs = Pop.decs();
            finalObj = obj;
            finalQ = Q;
            finalSim = Sim;
            finalIsValid = isValid;
        end

        if isempty(finalObj)
            warning('All snapshots empty in %s (SOLUTION class likely not on path when loading).', files(i).name);
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

        bestQ(i) = max(finalQ(finalIsValid));
        bestS(i) = max(finalSim(finalIsValid));
        simEq1Count(i) = sum(abs(finalSim(finalIsValid) - 1.0) < 1e-9);

        % Near-success diagnostics in the final snapshot
        maskSim = finalIsValid & (finalSim >= treS);
        if any(maskSim)
            maxQ_at_sim_ge_treS(i) = max(finalQ(maskSim));
        end
        maskQ = finalIsValid & (finalQ >= treQ);
        if any(maskQ)
            maxS_at_q_ge_treQ(i) = max(finalSim(maskQ));
        end
        count_sim_ge_treS(i) = sum(maskSim);
        count_q_ge_treQ(i) = sum(maskQ);
        maskBoth = maskSim & maskQ;
        count_both(i) = sum(maskBoth);

        maskLeadLike = finalIsValid & (abs(finalSim - 1.0) < 1e-9);
        if any(maskLeadLike)
            leadLikeQ(i) = max(finalQ(maskLeadLike));
        end

        successFinal(i) = any((finalQ(finalIsValid) >= treQ) & (finalSim(finalIsValid) >= treS));
        hvFinal(i) = hv_ref_2d(finalObj(finalIsValid,:), ref);

        % Collect final-population decision variables for SMILES decoding
        if saveDecs && ~isempty(finalDecs)
            [nPop, nDec] = size(finalDecs);
            for p = 1:nPop
                row = struct();
                row.mol_id = runNums(i) - 1;
                row.run    = runNums(i);
                row.pop_idx = p - 1;
                row.dec    = finalDecs(p, :);
                row.obj1   = finalObj(p, 1);
                row.obj2   = finalObj(p, 2);
                allDecsRows{end+1} = row; %#ok<AGROW>
            end
            mapRow = struct();
            mapRow.mol_id   = runNums(i) - 1;
            mapRow.run      = runNums(i);
            mapRow.mat_file = files(i).name;
            allMappingRows{end+1} = mapRow; %#ok<AGROW>
        end
    end

    T = table(runNums, string({files.name})', nNonEmptySnaps, ...
        successFinal, successAny, hvFinal, hvMax, ...
        bestQ, bestS, validFrac, simEq1Count, ...
        leadLikeQ, maxQ_at_sim_ge_treS, maxS_at_q_ge_treQ, count_sim_ge_treS, count_q_ge_treQ, count_both, ...
        'VariableNames', {'run', 'file', 'nSnaps', ...
        'successFinal', 'successAny', 'hvFinal', 'hvMax', ...
        'bestQED', 'bestSim', 'validFrac', 'simEq1Count', ...
        'leadLikeQED', 'maxQED_at_sim_ge_thr', 'maxSim_at_qed_ge_thr', 'countSim_ge_thr', 'countQED_ge_thr', 'countBoth'});

    summary = struct();
    summary.nRuns = N;
    summary.srCountFinal = sum(successFinal & ~isnan(hvFinal));
    summary.srRateFinal = summary.srCountFinal / N;
    summary.srCountAny = sum(successAny & ~isnan(hvMax));
    summary.srRateAny = summary.srCountAny / N;
    summary.hvFinalMean = mean(hvFinal(~isnan(hvFinal)));
    summary.hvFinalStd = std(hvFinal(~isnan(hvFinal)));
    summary.hvMaxMean = mean(hvMax(~isnan(hvMax)));
    summary.hvMaxStd = std(hvMax(~isnan(hvMax)));

    fprintf('\nOverall summary:\n');
    print_summary(summary, N, hvFinal, hvMax);

    stamp = datestr(now, 'yyyymmdd_HHMMSS');
    if writeFiles
        perRunPath = fullfile(outputDir, ['task1_metrics_per_run_' stamp '.csv']);
        summaryPath = fullfile(outputDir, ['task1_metrics_summary_' stamp '.txt']);
        writetable(T, perRunPath);
        write_summary_text(summaryPath, summary, N, hvFinal, hvMax, dataDir, char(opts.Pattern), treQ, treS, ref, perRunPath);
        fprintf('\nSaved outputs:\n');
        fprintf('  Per-run CSV : %s\n', perRunPath);
        fprintf('  Summary TXT : %s\n', summaryPath);

        % Save final-population decision variables for SMILES decoding
        if saveDecs && ~isempty(allDecsRows)
            decsPath = fullfile(outputDir, ['final_population_decs_' stamp '.csv']);
            fidDecs = fopen(decsPath, 'w');
            if fidDecs >= 0
                nDec = numel(allDecsRows{1}.dec);
                fprintf(fidDecs, 'mol_id,run,pop_idx');
                for d = 1:nDec
                    fprintf(fidDecs, ',dec_%d', d);
                end
                fprintf(fidDecs, ',obj1,obj2\n');
                for r = 1:numel(allDecsRows)
                    row = allDecsRows{r};
                    fprintf(fidDecs, '%d,%d,%d', row.mol_id, row.run, row.pop_idx);
                    fprintf(fidDecs, ',%.8g', row.dec);
                    fprintf(fidDecs, ',%.8g,%.8g\n', row.obj1, row.obj2);
                end
                fclose(fidDecs);
                fprintf('  Final-pop decs CSV : %s  (%d rows, %d dec dims)\n', decsPath, numel(allDecsRows), nDec);
            end

            mapPath = fullfile(outputDir, ['mol_id_to_mat_mapping_' stamp '.csv']);
            fidMap = fopen(mapPath, 'w');
            if fidMap >= 0
                fprintf(fidMap, 'mol_id,run,mat_file\n');
                for r = 1:numel(allMappingRows)
                    fprintf(fidMap, '%d,%d,%s\n', allMappingRows{r}.mol_id, allMappingRows{r}.run, allMappingRows{r}.mat_file);
                end
                fclose(fidMap);
                fprintf('  Mol-ID mapping   : %s  (%d runs)\n', mapPath, numel(allMappingRows));
            end
        end
    end

    if should_print_per_run(opts.ShowPerRun, N, opts.MaxPrintRows)
        print_per_run_table(T);
    else
        fprintf('\nPer-run table has %d rows, so it is not printed to the terminal by default.\n', N);
        fprintf('Use ShowPerRun=true to force terminal printing, or read the saved CSV above.\n');
    end

    fprintf('\nFinal summary (repeated so it stays visible in AutoDL logs):\n');
    print_summary(summary, N, hvFinal, hvMax);
end

function print_summary(summary, N, hvFinal, hvMax)
    fprintf('  SR (final): %d / %d (%.3f)\n', summary.srCountFinal, N, summary.srRateFinal);
    fprintf('  SR (any)  : %d / %d (%.3f)\n', summary.srCountAny, N, summary.srRateAny);
    fprintf('  HV (final): mean=%.6f, std=%.6f (N=%d)\n', summary.hvFinalMean, summary.hvFinalStd, sum(~isnan(hvFinal)));
    fprintf('  HV (max)  : mean=%.6f, std=%.6f (N=%d)\n', summary.hvMaxMean, summary.hvMaxStd, sum(~isnan(hvMax)));
end

function tf = should_print_per_run(showPerRun, N, maxPrintRows)
    if islogical(showPerRun) || isnumeric(showPerRun)
        tf = logical(showPerRun);
        return;
    end

    value = lower(strtrim(char(showPerRun)));
    if any(strcmp(value, {'true', 'on', 'yes', '1'}))
        tf = true;
    elseif any(strcmp(value, {'false', 'off', 'no', '0'}))
        tf = false;
    elseif strcmp(value, 'auto')
        tf = N <= maxPrintRows;
    else
        error('ShowPerRun must be true, false, or ''auto''.');
    end
end

function print_per_run_table(T)
    fprintf('\nPer-run (sorted by run id):\n');
    fprintf('%-6s %-5s %-5s %-7s %-12s %-12s %-10s %-10s %-10s %-8s %-10s %-10s\n', ...
        'run', 'fin', 'any', 'snaps', 'hvFinal', 'hvMax', 'bestQED', 'bestSim', 'validFrac', 'sim==1', 'Q@sim>=thr', 'sim@Q>=thr');
    for i = 1:height(T)
        fprintf('%-6d %-5d %-5d %-7d %-12.6f %-12.6f %-10.4f %-10.4f %-10.3f %-8d %-10.4f %-10.4f\n', ...
            T.run(i), int32(T.successFinal(i)), int32(T.successAny(i)), int32(T.nSnaps(i)), ...
            T.hvFinal(i), T.hvMax(i), T.bestQED(i), T.bestSim(i), T.validFrac(i), int32(T.simEq1Count(i)), ...
            T.maxQED_at_sim_ge_thr(i), T.maxSim_at_qed_ge_thr(i));
    end
end

function write_summary_text(summaryPath, summary, N, hvFinal, hvMax, dataDir, pattern, treQ, treS, ref, perRunPath)
    fid = fopen(summaryPath, 'w');
    if fid < 0
        warning('Cannot write summary file: %s', summaryPath);
        return;
    end

    cleaner = onCleanup(@()fclose(fid));
    fprintf(fid, 'MOMO Task1 metrics summary\n');
    fprintf(fid, 'Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, 'DataDir: %s\n', dataDir);
    fprintf(fid, 'Pattern: %s\n', pattern);
    fprintf(fid, 'Thresholds: QED>=%.3f, SIM>=%.3f | HV ref=[%.3f, %.3f]\n\n', treQ, treS, ref(1), ref(2));
    fprintf(fid, 'SR (final): %d / %d (%.3f)\n', summary.srCountFinal, N, summary.srRateFinal);
    fprintf(fid, 'SR (any)  : %d / %d (%.3f)\n', summary.srCountAny, N, summary.srRateAny);
    fprintf(fid, 'HV (final): mean=%.6f, std=%.6f (N=%d)\n', summary.hvFinalMean, summary.hvFinalStd, sum(~isnan(hvFinal)));
    fprintf(fid, 'HV (max)  : mean=%.6f, std=%.6f (N=%d)\n', summary.hvMaxMean, summary.hvMaxStd, sum(~isnan(hvMax)));
    fprintf(fid, '\nPer-run CSV: %s\n', perRunPath);
    clear cleaner;
end

function hv = hv_ref_2d(points, ref)
% Exact dominated HV in 2D for minimization problems.
% points: Nx2 objective vectors (should satisfy points(:,j) <= ref(j)).
% ref   : 1x2 reference point.

    if isempty(points)
        hv = 0;
        return;
    end

    % Keep only points that dominate the reference point in minimization sense.
    mask = all(points <= ref, 2) & all(isfinite(points), 2);
    points = points(mask, :);
    if isempty(points)
        hv = 0;
        return;
    end

    % Remove duplicates to stabilize sorting.
    points = unique(points, 'rows');

    % Nondominated filtering (2D): sort by f1 asc, keep strictly improving f2.
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

    % Compute area as union of rectangles to ref.
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
