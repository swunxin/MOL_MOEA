function out = State(pop_or_cmd, pi_prev, cfg)
% 7维顺序：
% [1 NDR_prev2cur;
%  2 C_BA;
%  3 C_AB;
%  4 pi_prev;
%  5 PF1_ratio;
%  6 s4_hist_gap;   % 论文 s4
%  7 s5_diversity]  % 论文 s5：全局多样性
%
% cfg.feature_mask (default=[1 0 0 1 1 1 1])  % 7位
% cfg.current_fitness / fitness_handle / fitness_weights / objective_index
% cfg.minimize (default=true): 若 fi 是最大化意义，置 false（内部取负转最小化）
% cfg.s5_normalize (default=true): 先对 objs 做列归一化再算 std 均值（更稳）

    if nargin < 3 || isempty(cfg), cfg = struct(); end
    use_cache_only = getfield_def(cfg,'use_cache_only', true);
    zero_on_miss   = getfield_def(cfg,'zero_on_miss',   true);
    fmask          = logical(getfield_def(cfg,'feature_mask', [1 0 0 1 1 1 1]));
    do_minimize_f  = getfield_def(cfg,'minimize', true);
    s5_norm        = getfield_def(cfg,'s5_normalize', true);

    persistent prevObjs nextCache histBestFit

    % —— 控制命令 ——
    if (ischar(pop_or_cmd) || isstring(pop_or_cmd))
        cmd = lower(string(pop_or_cmd));
        if cmd == "reset"
            prevObjs    = [];
            nextCache   = [];
            histBestFit = [];
            out = []; return;
        elseif cmd == "push_cache"
            nextCache = pi_prev;   % 第二参数位置复用为 cache struct
            out = []; return;
        end
    end

    % —— 正常构造状态 ——
    objs = pop_or_cmd.objs;   % 注意：默认 objs 已经是“最小化形式”（Python 做过转换）

    % 1) 先拿 NDR/C（优先吃 push_cache）
    haveCache = ~isempty(nextCache) && isstruct(nextCache) ...
              && getfield_def(nextCache,'have',false);

    if haveCache
        ndr = getfield_def(nextCache,'ndr_prev2cur', 0);
        cba = getfield_def(nextCache,'c_ba',         0);
        cab = getfield_def(nextCache,'c_ab',         0);
        nextCache = [];
    else
        if use_cache_only
            if zero_on_miss || isempty(prevObjs)
                ndr = 0; cba = 0; cab = 0;
            else
                [cba, cab, ndr] = cov_ndr_pair(prevObjs, objs);
            end
        else
            if isempty(prevObjs)
                ndr = 0; cba = 0; cab = 0;
            else
                [cba, cab, ndr] = cov_ndr_pair(prevObjs, objs);
            end
        end
    end

    % 2) PF1/N（最小化）
    pf1_ratio = pf1_ratio_of(objs);

    % 3) 标量适应度 fi（用于 s4）
    if isfield(cfg,'current_fitness') && ~isempty(cfg.current_fitness)
        fi = cfg.current_fitness(:);
    else
        fi = scalar_fitness_of(objs, cfg);
    end
    if ~do_minimize_f
        fi = -fi;  % fi 若是“越大越好”，转成最小化
    end

    % s4: 历史最优收敛差距（论文 s4）
    curBest = min(fi);
    if isempty(histBestFit)
        histBestFit = curBest;
    else
        histBestFit = min(histBestFit, curBest);
    end
    s4 = mean(fi - histBestFit);

    % s5: 全局多样性（论文 s5）：各目标维 std 的平均
    if size(objs,1) <= 1
        s5 = 0;
    else
        if s5_norm
            minv = min(objs, [], 1);
            den  = max(max(objs, [], 1) - minv, eps);
            objsN = (objs - minv) ./ den;
            s5 = mean(std(objsN, 0, 1));
        else
            s5 = mean(std(objs, 0, 1));
        end
    end

    % 记录当前代（用于回退）
    prevObjs = objs;

    % 上一步动作概率
    if nargin < 2 || isempty(pi_prev) || ~isscalar(pi_prev)
        pi_feat = 0;
    else
        pi_feat = pi_prev;
    end

    feat = [ndr; cba; cab; pi_feat; pf1_ratio; s4; s5];

    if numel(fmask) ~= numel(feat)
        error('cfg.feature_mask length (%d) must match feature length (%d).', numel(fmask), numel(feat));
    end

    out = feat(fmask);
end

function fi = scalar_fitness_of(Objs, cfg)
    N = size(Objs,1);
    if N == 0, fi = 0; return; end

    if isfield(cfg,'fitness_handle') && ~isempty(cfg.fitness_handle)
        f = cfg.fitness_handle;
        fi = arrayfun(@(i) f(Objs(i,:)), (1:N).');
    elseif isfield(cfg,'fitness_weights') && ~isempty(cfg.fitness_weights)
        w = cfg.fitness_weights(:);
        if size(Objs,2) ~= numel(w)
            error('fitness_weights size mismatch with number of objectives.');
        end
        fi = Objs * w;
    else
        idx = getfield_def(cfg,'objective_index', 1);
        fi = Objs(:, idx);
    end
end

function [C_BA, C_AB, NDR] = cov_ndr_pair(A_objs, B_objs)
    Combined = [A_objs; B_objs];
    minv  = min(Combined, [], 1);
    range = max(max(Combined, [], 1) - minv, eps);
    A = (A_objs - minv) ./ range;
    B = (B_objs - minv) ./ range;

    C_BA = coverage_one(B, A);
    C_AB = coverage_one(A, B);
    NDR  = C_BA - C_AB;
end

function C = coverage_one(S1, S2)
    if isempty(S2), C = 0; return; end
    n2 = size(S2,1);
    covered = false(n2,1);
    for j = 1:n2
        d1 = all(S1 <= S2(j,:), 2);
        d2 = any(S1 <  S2(j,:), 2);
        if any(d1 & d2), covered(j) = true; end
    end
    C = mean(covered);
end

function r = pf1_ratio_of(Objs)
    N = size(Objs,1);
    if N == 0, r = 0; return; end

    if exist('NDSort','file') == 2
        [FrontNo, ~] = NDSort(Objs, N);
    elseif exist('NDsort','file') == 2
        [FrontNo, ~] = NDsort(Objs, N);
    else
        FrontNo = ones(N,1);
        for i = 1:N
            for j = 1:N
                if j ~= i && all(Objs(j,:) <= Objs(i,:)) && any(Objs(j,:) < Objs(i,:))
                    FrontNo(i) = 2;
                    break;
                end
            end
        end
    end

    r = sum(FrontNo == 1) / N;
end

function v = getfield_def(s, name, def)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name)), v = s.(name); else, v = def; end
end
