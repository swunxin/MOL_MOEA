function visualize_momo_taskB_runs()
% Visualize MOMO-style taskB runs (QED + similarity) saved by PlatEMO.
%
% It loads all .mat files under Data/ANSGAIII/, extracts the final
% population objectives, converts them back to (QED, similarity) by
% negating the minimization objectives, and plots:
%   1) all final populations (QED vs sim)
%   2) per-run best point (max QED, max sim)
% It also prints a quick per-run summary to help verify the pipeline.

    root = fileparts(mfilename('fullpath'));
    cd(root);
    addpath(genpath(root));

    dataDir = fullfile(root, 'Data', 'ANSGAIII');
    if exist(dataDir, 'dir') ~= 7
        error('Data directory not found: %s', dataDir);
    end

    files = dir(fullfile(dataDir, 'ANSGAIII_DDProblem1_M2_D256_*.mat'));
    if isempty(files)
        error('No result files found in: %s', dataDir);
    end

    allQ = [];
    allS = [];
    allRun = [];

    bestQ = nan(numel(files), 1);
    bestS = nan(numel(files), 1);
    validFrac = nan(numel(files), 1);

    fprintf('Found %d runs in %s\n', numel(files), dataDir);

    for i = 1:numel(files)
        path = fullfile(files(i).folder, files(i).name);
        S = load(path, 'result');
        if ~isfield(S, 'result') || isempty(S.result) || size(S.result,2) < 2
            warning('Missing/invalid result in %s', files(i).name);
            continue;
        end

        % Find the last non-empty population snapshot
        idx = [];
        for r = size(S.result,1):-1:1
            if ~isempty(S.result{r,2})
                idx = r;
                break;
            end
        end
        if isempty(idx)
            warning('All snapshots empty in %s (SOLUTION class likely not on path when loading).', files(i).name);
            continue;
        end

        Pop = S.result{idx,2};

        % Robust objective extraction for SOLUTION arrays
        % (objs is a method on SOLUTION)
        obj = Pop.objs();

        if size(obj,2) ~= 2
            warning('Expected M=2 objectives but got %d in %s', size(obj,2), files(i).name);
            continue;
        end

        % Convert minimization objectives back to the intuitive maximization view
        Q = -obj(:,1);    % QED in [0,1]
        Sim = -obj(:,2);  % similarity in [0,1]

        % Track validity (invalid molecules are penalized as obj_qed=1, obj_sim=0)
        isValid = (obj(:,1) <= 0) & (obj(:,2) <= 0);

        allQ = [allQ; Q];
        allS = [allS; Sim];
        allRun = [allRun; repmat(i, numel(Q), 1)];

        bestQ(i) = max(Q(isValid));
        bestS(i) = max(Sim(isValid));
        validFrac(i) = mean(isValid);

        if isnan(bestQ(i)); bestQ(i) = max(Q); end
        if isnan(bestS(i)); bestS(i) = max(Sim); end
    end

    % Plot all final populations
    figure('Name','Final populations (QED vs similarity)','NumberTitle','off');
    scatter(allQ, allS, 8, allRun, 'filled');
    xlabel('QED (higher is better)');
    ylabel('Similarity to lead (higher is better)');
    title('ANSGAIII final populations across runs');
    grid on;
    colormap(parula);
    cb = colorbar;
    cb.Label.String = 'Run index (file order)';

    % Plot per-run best points
    figure('Name','Per-run best summary','NumberTitle','off');
    scatter(bestQ, bestS, 30, 'filled');
    xlabel('Best QED in run');
    ylabel('Best similarity in run');
    title('Best-of-run summary (sanity check)');
    grid on;

    % Print summary table
    fprintf('\nPer-run summary (file order):\n');
    fprintf('%-4s %-30s %-10s %-10s %-10s\n', 'i', 'file', 'bestQED', 'bestSim', 'validFrac');
    for i = 1:numel(files)
        fprintf('%-4d %-30s %-10.4f %-10.4f %-10.3f\n', i, files(i).name, bestQ(i), bestS(i), validFrac(i));
    end

    fprintf('\nSanity expectations:\n');
    fprintf('- QED and similarity should mostly lie in [0,1].\n');
    fprintf('- validFrac close to 1.0 means most decoded molecules are valid.\n');
    fprintf('- If everything is empty, ensure PlatEMO root is on path so SOLUTION loads.\n');
end
