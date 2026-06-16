function New_POP = Pop_Pool(Population_A, Population_B, Offspring, Problem)
    tol_rel = 1e-6;

    if isempty(Population_A)
        PF1_A = Population_A;
    else
        [FrontNo_A, ~] = NDSort(Population_A.objs, numel(Population_A));
        PF1_A = Population_A(FrontNo_A == 1);
    end

    if isempty(Population_B)
        PF1_B = Population_B;
    else
        [FrontNo_B, ~] = NDSort(Population_B.objs, numel(Population_B));
        PF1_B = Population_B(FrontNo_B == 1);
    end

    PF1_A = unique_pop_by_objs_stable_scaled(PF1_A, tol_rel);
    PF1_B = unique_pop_by_objs_stable_scaled(PF1_B, tol_rel);

    PF1_union  = [PF1_A, PF1_B];
    PF1_unique = unique_pop_by_objs_stable_scaled(PF1_union, tol_rel);
    numA = numel(PF1_A);
    PF1_A = PF1_unique(1:min(numA,numel(PF1_unique)));
    PF1_B = PF1_unique(min(numA,numel(PF1_unique))+1:end);

    if numel(PF1_A) >= Problem.N
        F_A  = cat(1, PF1_A.objs);
        CD_A = crowding_distance_min(F_A);
        [~, ordA] = sort(CD_A, 'descend');
        New_POP = PF1_A(ordA(1:Problem.N));
        B_pool  = PF1_B([]);
    else
        New_POP = PF1_A;
        B_pool  = PF1_B;
    end

    if numel(New_POP) < Problem.N && ~isempty(Offspring)
        M = floor(numel(Offspring)/2);
        if M > 0
            NewKidsAll = Offspring(1:M);
            merged = unique_pop_by_objs_stable_scaled([New_POP, NewKidsAll], tol_rel);
            RemKids = merged(numel(New_POP)+1:end);

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
                    idxF  = find(FrontNo == f);
                    inRem = idxF(idxF > base);
                    if isempty(inRem)
                        f = f + 1;
                        continue;
                    end

                    space = Need - numel(add);
                    if numel(inRem) <= space
                        add = [add; inRem(:)]; %#ok<AGROW>
                    else
                        CD = crowding_distance_min(F(idxF,:));
                        [tf,loc] = ismember(inRem, idxF);
                        inRem = inRem(tf);
                        loc   = loc(tf);
                        if isempty(loc)
                            f = f + 1;
                            continue;
                        end
                        [~,ord] = sort(CD(loc), 'descend');
                        pick = inRem(ord(1:space));
                        add = [add; pick(:)]; %#ok<AGROW>
                    end
                    f = f + 1;
                end

                if ~isempty(add)
                    New_POP = [New_POP, CandAll(add)];
                end
            end

            if numel(New_POP) < Problem.N && ~isempty(NewKidsAll)
                FillKids = remove_selected(NewKidsAll, New_POP);
                fill = Problem.N - numel(New_POP);
                if ~isempty(FillKids) && fill > 0
                    idx = randperm(numel(FillKids), min(fill, numel(FillKids)));
                    New_POP = [New_POP, FillKids(idx)];
                end
            end
        end
    end

    if ~isempty(B_pool) && numel(New_POP) < Problem.N
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
            New_POP(end+1) = New_POP(randi(numel(New_POP))); %#ok<AGROW>
        end
    end
end

function Puniq = unique_pop_by_objs_stable_scaled(P, tol_rel)
    if isempty(P)
        Puniq = P;
        return;
    end

    F = cat(1, P.objs);
    fmin = min(F,[],1);
    fmax = max(F,[],1);
    span = fmax - fmin;
    span(span == 0) = 1;
    Fn = (F - fmin) ./ span;

    K = size(Fn,1);
    keep = false(K,1);
    keptF = [];
    for i = 1:K
        fi = Fn(i,:);
        if isempty(keptF)
            keep(i) = true;
            keptF = fi;
        else
            d = max(abs(keptF - fi), [], 2);
            if min(d) > tol_rel
                keep(i) = true;
                keptF = [keptF; fi]; %#ok<AGROW>
            end
        end
    end
    Puniq = P(keep);
end

function CD = crowding_distance_min(F)
    [N,M] = size(F);
    if N == 0
        CD = [];
        return;
    end

    fmin = min(F,[],1);
    fmax = max(F,[],1);
    span = fmax - fmin;
    span(span == 0) = 1;
    Fn = (F - fmin) ./ span;

    CD = zeros(N,1);
    for j = 1:M
        [fj, idx] = sort(Fn(:,j), 'ascend');
        CD(idx(1)) = inf;
        CD(idx(end)) = inf;
        if N > 2
            delta = fj(3:end) - fj(1:end-2);
            CD(idx(2:end-1)) = CD(idx(2:end-1)) + delta;
        end
    end
end

function PoolOut = remove_selected(PoolIn, Selected)
    if isempty(PoolIn) || isempty(Selected)
        PoolOut = PoolIn;
        return;
    end

    mask = true(1, numel(PoolIn));
    for i = 1:numel(PoolIn)
        for j = 1:numel(Selected)
            if isequal(PoolIn(i), Selected(j))
                mask(i) = false;
                break;
            end
        end
    end
    PoolOut = PoolIn(mask);
end
