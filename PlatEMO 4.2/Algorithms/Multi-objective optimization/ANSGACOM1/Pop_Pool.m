function New_POP = Pop_Pool(Population_A, Population_B, Offspring, Problem)
    % ===== Step 1: NDSort =====
    [FrontNo_A, ~] = NDSort(Population_A.objs, numel(Population_A));
    [FrontNo_B, ~] = NDSort(Population_B.objs, numel(Population_B));

    % ===== Step 2: PF=1 去重 + 组装（A 优先）=====
    New_POP = [];
    PF1_A = Population_A(FrontNo_A == 1);
    PF1_B = Population_B(FrontNo_B == 1);

    tol_rel = 1e-6;   % 归一后∞范数去重阈值
    PF1_A = unique_pop_by_objs_stable_scaled(PF1_A, tol_rel);
    PF1_B = unique_pop_by_objs_stable_scaled(PF1_B, tol_rel);

    PF1_union  = [PF1_A, PF1_B];
    PF1_unique = unique_pop_by_objs_stable_scaled(PF1_union, tol_rel);
    nA = min(numel(PF1_A), numel(PF1_unique));
    PF1_A = PF1_unique(1:nA);
    PF1_B = PF1_unique(nA+1:end);

    if numel(PF1_A) >= Problem.N
        F_A  = cat(1, PF1_A.objs);
        CD_A = crowding_distance_min(F_A);
        [~, ordA] = sort(CD_A, 'descend');
        New_POP = PF1_A(ordA(1:Problem.N));
        B_pool  = [];
    else
        New_POP = PF1_A;
        B_pool  = PF1_B;
    end

    %% ===== Step 3: 只从 Offspring 的前 M 个新点补齐 =====
    if numel(New_POP) < Problem.N && ~isempty(Offspring)
        M = floor(numel(Offspring)/2);
        if M > 0
            NewKidsAll = Offspring(1:M);

            % (1) 去掉与已选集合“近重复”的孩子（优先保留已选）
            merged  = unique_pop_by_objs_stable_scaled([New_POP, NewKidsAll], tol_rel);
            RemKids = merged(numel(New_POP)+1:end);

            % (2) 只从 RemKids 里补：先做 NDSort，再在每个前沿里按 CD 择优
            if ~isempty(RemKids)
                Need    = Problem.N - numel(New_POP);
                CandAll = [New_POP, RemKids];
                F       = cat(1, CandAll.objs);
                [FrontNo, ~] = NDSort(F, size(F,1));

                add = zeros(0,1);
                f   = 1;
                Fmax = max(FrontNo);
                base = numel(New_POP);

                while numel(add) < Need && f <= Fmax
                    idxF  = find(FrontNo==f);           % 该前沿在 CandAll 的索引
                    inRem = idxF(idxF > base);          % 仅保留 RemKids 的那些

                    if isempty(inRem)
                        f = f + 1; 
                        continue;
                    end

                    space = Need - numel(add);
                    if numel(inRem) <= space
                        add = [add; inRem(:)];
                    else
                        % —— 稳健处理：防止 loc==0 —— %
                        CD       = crowding_distance_min(F(idxF,:));
                        [tf,loc] = ismember(inRem, idxF);   % loc 为 inRem 在 idxF 中的位置（可能有 0）
                        inRem    = inRem(tf);               % 只保留真子集
                        loc      = loc(tf);
                        if isempty(loc)                     % 极端保护
                            f = f + 1; 
                            continue;
                        end
                        [~,ord] = sort(CD(loc), 'descend');
                        pick    = inRem(ord(1:space));
                        add     = [add; pick(:)];
                    end
                    f = f + 1;
                end

                if ~isempty(add)
                    New_POP = [New_POP, CandAll(add)];
                end
            end

            % (3) 兜底：若仍不足 N，从 NewKidsAll 随机补
            if numel(New_POP) < Problem.N && ~isempty(NewKidsAll)
                fill = Problem.N - numel(New_POP);
                idx  = randperm(numel(NewKidsAll), min(fill, numel(NewKidsAll)));
                New_POP = [New_POP, NewKidsAll(idx)];
            end
        end
    end

    % ===== 兜底：若 Offspring 补完仍不足 N，用 B_pool 补齐 =====
    if exist('B_pool','var') && ~isempty(B_pool) && numel(New_POP) < Problem.N
        merged  = unique_pop_by_objs_stable_scaled([New_POP, B_pool], tol_rel);
        B_pool2 = merged(numel(New_POP)+1:end);
        if ~isempty(B_pool2)
            fill = Problem.N - numel(New_POP);
            if numel(B_pool2) > fill
                F_B  = cat(1, B_pool2.objs);
                CD_B = crowding_distance_min(F_B);
                [~,ord] = sort(CD_B, 'descend');
                New_POP = [New_POP, B_pool2(ord(1:fill))];
            else
                New_POP = [New_POP, B_pool2];
            end
        end
    end

    % ===== 最终兜底：必须凑满 N（防 SOM 邻接越界）=====
    if numel(New_POP) < Problem.N
        Pool = [Population_A, Population_B];
        Pool = remove_selected(Pool, New_POP);
        need = Problem.N - numel(New_POP);
        if ~isempty(Pool)
            k = min(need, numel(Pool));
            if k > 0
                idx = randperm(numel(Pool), k);
                New_POP = [New_POP, Pool(idx)];
            end
        end
        while numel(New_POP) < Problem.N && ~isempty(New_POP)
            New_POP(end+1) = New_POP(randi(numel(New_POP)));
        end
    end
end

%% ======= 工具函数们 =======
function Puniq = unique_pop_by_objs_stable_scaled(P, tol_rel)
    if isempty(P), Puniq = P; return; end
    F = cat(1, P.objs);
    fmin = min(F,[],1); fmax = max(F,[],1);
    span = fmax - fmin; span(span==0) = 1;
    Fn = (F - fmin) ./ span;

    K = size(Fn,1); keep = false(K,1); keptF = [];
    for i = 1:K
        fi = Fn(i,:);
        if isempty(keptF)
            keep(i) = true; keptF = fi;
        else
            d = max(abs(keptF - fi), [], 2);
            if min(d) > tol_rel
                keep(i) = true; keptF = [keptF; fi]; %#ok<AGROW>
            end
        end
    end
    Puniq = P(keep);
end

function CD = crowding_distance_min(F)
    [N,M] = size(F);
    if N==0, CD=[]; return; end
    fmin = min(F,[],1); fmax = max(F,[],1);
    span = fmax - fmin; span(span==0)=1;
    Fn = (F - fmin) ./ span;

    CD = zeros(N,1);
    for j = 1:M
        [fj, idx] = sort(Fn(:,j), 'ascend');
        CD(idx(1))   = inf;
        CD(idx(end)) = inf;
        if N > 2
            delta = fj(3:end) - fj(1:end-2);
            CD(idx(2:end-1)) = CD(idx(2:end-1)) + delta;
        end
    end
end

function PoolOut = remove_selected(PoolIn, Selected)
    if isempty(PoolIn) || isempty(Selected)
        PoolOut = PoolIn; return;
    end
    mask = true(1, numel(PoolIn));
    for i = 1:numel(PoolIn)
        for j = 1:numel(Selected)
            if isequal(PoolIn(i), Selected(j))
                mask(i) = false; break;
            end
        end
    end
    PoolOut = PoolIn(mask);
end
