% Read the final non-empty population snapshot from a PlatEMO .mat result
% file (M=2 case: QED + Docking), and print basic diagnostics.
%
% Usage:
%   1) Open MATLAB and run this script (it auto-adds PlatEMO root to path)
%   2) Optionally edit dataDir / pattern below

clear; clc;

% Mirror compute_momo_task1_metrics_from_platemo.m path style:
% root = PlatEMO root, and dataDir is user-editable.
scriptDir = fileparts(mfilename('fullpath'));

% Robustly find PlatEMO root by searching upward for platemo.m + Problems/SOLUTION.m
root = '';
probe = scriptDir;
for k = 1:8
    if exist(fullfile(probe, 'platemo.m'), 'file') == 2 && ...
       exist(fullfile(probe, 'Problems', 'SOLUTION.m'), 'file') == 2
        root = probe;
        break;
    end
    parent = fileparts(probe);
    if strcmp(parent, probe)
        break;
    end
    probe = parent;
end

if isempty(root)
    error(['Cannot auto-locate PlatEMO root from scriptDir=%s. ' ...
           'Please set variable root manually in this script.'], scriptDir);
end

cd(root);
addpath(genpath(root));
fprintf('PlatEMO root added to path: %s\n', root);

% User-configurable path settings (same style as compute_momo_task1_metrics_from_platemo.m)
% This reader script is placed under PlatEMO root in your workflow.
% Use a fixed result directory under root (no auto-detection).
dataDir = fullfile(root, 'Data', 'LMOCSO');
pattern = 'LMOCSO_DDProblem1_M2_D*_*.mat';

if exist(dataDir, 'dir') ~= 7
    error('Data directory not found: %s', dataDir);
end

files = dir(fullfile(dataDir, pattern));
if isempty(files)
    error('No result files found in: %s (pattern: %s)', dataDir, pattern);
end

% If multiple runs exist, pick the latest by trailing run number
runNums = nan(numel(files), 1);
for i = 1:numel(files)
    tok = regexp(files(i).name, '_(\d+)\.mat$', 'tokens', 'once');
    if ~isempty(tok)
        runNums(i) = str2double(tok{1});
    end
end
[~, order] = sort(runNums);
files = files(order);
mat_path = fullfile(files(end).folder, files(end).name);

S = load(mat_path, 'result', 'metric');

if ~isfield(S, 'result') || isempty(S.result) || size(S.result,2) < 2
    error('Invalid result format in %s', mat_path);
end

% Find the last non-empty snapshot (final population)
idx = [];
for r = size(S.result,1):-1:1
    if ~isempty(S.result{r,2})
        idx = r;
        break;
    end
end

if isempty(idx)
    error(['All snapshots are empty. This usually means SOLUTION.m was not on MATLAB path when loading. ' ...
           'Auto-added PlatEMO root was: %s'], root);
end

FE  = S.result{idx,1};
Pop = S.result{idx,2};   % SOLUTION array

Obj = Pop.objs();        % N x 2  ([-QED, docking])
Dec = Pop.decs();        % N x 256
Con = Pop.cons();        % usually empty/zero-column

QED_val  = -Obj(:,1);
Dock_val =  Obj(:,2);

fprintf('Loaded file: %s\n', mat_path);
if isfield(S, 'metric') && isfield(S.metric, 'runtime')
    fprintf('Runtime: %.3f s\n', S.metric.runtime);
end
fprintf('Final snapshot row: %d\n', idx);
fprintf('Final FE: %d\n', FE);
fprintf('Population size: %d\n', size(Obj,1));
fprintf('Decision dim: %d\n', size(Dec,2));
fprintf('Objective dim: %d\n', size(Obj,2));
fprintf('Constraint cols: %d\n', size(Con,2));

fprintf('\nQED stats (converted from -QED): min=%.4f, max=%.4f, mean=%.4f\n', ...
    min(QED_val), max(QED_val), mean(QED_val));
fprintf('Dock stats: min=%.4f, max=%.4f, mean=%.4f\n', ...
    min(Dock_val), max(Dock_val), mean(Dock_val));

disp('First 5 objective rows [ -QED, Docking ]:');
disp(Obj(1:min(5,end), :));

% Optional exports for downstream filtering (comment out if not needed)
% writematrix(Obj, strrep(mat_path, '.mat', '_final_obj.csv'));
% writematrix(Dec, strrep(mat_path, '.mat', '_final_dec.csv'));
