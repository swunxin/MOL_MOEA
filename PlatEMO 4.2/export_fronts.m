% export_fronts.m — 导出指定 lead 的最终前沿 (qed,sim)，对比 ours vs FRCSO。
%   obj = [-qed, -sim]，所以 qed=-obj1, sim=-obj2。
%   选了几个 contested lead：FRCSO优势(2,48,69,139,191) + 我们优势(55) + 都失败(1)。
%   输出 front_compare.csv (algo,run,qed,sim)，推回本地分析/画图。
clear; clc;
scriptDir = fileparts(mfilename('fullpath'));
cd(scriptDir); addpath(genpath(scriptDir));

algos = {'MOL_MOEA_v10_bank','FRCSO_N100'};
runs  = [2 48 69 139 191 55 1];

rows = {};
for a = 1:numel(algos)
    for rr = runs
        f = fullfile(scriptDir,'Data',algos{a}, ...
            sprintf('%s_DDProblem1_M2_D_%d.mat',algos{a},rr));
        if ~isfile(f); fprintf('缺文件: %s\n', f); continue; end
        S = load(f);
        if ~isfield(S,'result'); fprintf('无 result: %s\n', f); continue; end
        R = S.result;
        % 取最后一个非空快照的 Population
        P = [];
        for k = size(R,1):-1:1
            if ~isempty(R{k,2}); P = R{k,2}; break; end
        end
        if isempty(P); continue; end
        if isobject(P); O = P.objs; else; O = cat(1, P.obj); end   % [-qed,-sim]
        for i = 1:size(O,1)
            rows(end+1,:) = {algos{a}, rr, -O(i,1), -O(i,2)}; %#ok<AGROW>
        end
    end
end

T = cell2table(rows,'VariableNames',{'algo','run','qed','sim'});
out = fullfile(scriptDir,'front_compare.csv');
writetable(T, out);
fprintf('写出: %s  (%d 行)\n', out, height(T));
% 顺手报每个 (algo,run) 在达标框 qed>=0.9 & sim>=0.4 内的点数
for a = 1:numel(algos)
    for rr = runs
        sub = T(strcmp(T.algo,algos{a}) & T.run==rr, :);
        if isempty(sub); continue; end
        nin = sum(sub.qed>=0.9 & sub.sim>=0.4);
        fprintf('  %-20s run%-4d N=%-3d 达标框内=%d  qed[%.3f,%.3f] sim[%.3f,%.3f]\n', ...
            algos{a}, rr, height(sub), nin, min(sub.qed),max(sub.qed),min(sub.sim),max(sub.sim));
    end
end
