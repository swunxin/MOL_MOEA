% Read the final non-empty population snapshot from a PlatEMO .mat result
% file for optimizer_orig.py M=6 case, and print objective diagnostics.
%
% optimizer_orig.py objective column order (as stored in PlatEMO result):
%   1) bioavailability_ma   (converted objective, minimize)
%   2) ClinTox             (converted objective, minimize)
%   3) LD50_Zhu            (converted objective, minimize)
%   4) solubility_aqsoldb  (converted objective, minimize)
%   5) SA_Score            (raw SA score, minimize)
%   6) Binding_Affinity    (docking score, minimize; more negative is better)
%
% Usage:
%   1) Open MATLAB and run this script (it auto-adds PlatEMO root to path)
%   2) Optionally edit dataDir / pattern below

clear; clc;

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

% User-configurable path settings
dataDir = fullfile(root, 'Data', 'NMPSO');
pattern = 'NMPSO_DDProblem1_M2_D*_*.mat';

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
Pop = S.result{idx,2};

Obj = Pop.objs();
Dec = Pop.decs();
Con = Pop.cons();

objNames = { ...
    'bioavailability_ma(conv)', ...
    'ClinTox(conv)', ...
    'LD50_Zhu(conv)', ...
    'solubility_aqsoldb(conv)', ...
    'SA_Score', ...
    'Binding_Affinity(dock)' ...
};

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

if size(Obj,2) ~= 6
    warning('Expected M=6, but file contains M=%d objectives.', size(Obj,2));
end

fprintf('\nObjective statistics (raw stored values in PlatEMO):\n');
for j = 1:size(Obj,2)
    if j <= numel(objNames)
        name = objNames{j};
    else
        name = sprintf('obj_%d', j);
    end
    col = Obj(:,j);
    fprintf('  [%d] %-28s min=% .6f   max=% .6f   mean=% .6f\n', ...
        j, name, min(col), max(col), mean(col));
end

if size(Obj,2) >= 6
    dockCol = Obj(:,6);
    fprintf('\nDocking column (obj6) quick check:\n');
    fprintf('  min=% .6f, max=% .6f, mean=% .6f\n', min(dockCol), max(dockCol), mean(dockCol));
    fprintf('  unique values among first 20 entries:\n');
    disp(unique(dockCol(1:min(20,end)))');
end

disp('First 5 objective rows:');
disp(Obj(1:min(5,end), :));

% Optional exports for downstream analysis (comment out if not needed)
% writematrix(Obj, strrep(mat_path, '.mat', '_final_obj.csv'));
% writematrix(Dec, strrep(mat_path, '.mat', '_final_dec.csv'));

