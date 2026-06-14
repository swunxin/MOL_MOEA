function MatingPool = SOMSelection(Population, Fitness, somCfg, Problem)
% CKD-style SOM selection in decision space, returning a MatingPool index list.

    Npop  = numel(Population);
    GridN = somCfg.GridN;
    if GridN < 1 || Npop < 1
        MatingPool = randi(Npop, 1, Npop);
        return;
    end

    if GridN < Npop
        idx = randperm(Npop, GridN);
        PopSel = Population(idx);
        FitSel = Fitness(idx);
    else
        idx = 1:Npop;
        PopSel = Population;
        FitSel = Fitness;
    end

    S = PopSel.decs;
    W = S;
    LDis = somCfg.LDis;
    B = somCfg.B;

    for s = 1:size(S,1)
        sigma  = somCfg.sigma0*(1-(Problem.FE+s)/Problem.maxFE);
        tau    = somCfg.tau0*(1-(Problem.FE+s)/Problem.maxFE);
        [~,u1] = min(pdist2(S(s,:),W));
        U      = LDis(u1,:) < sigma;
        if any(U)
            W(U,:) = W(U,:) + tau.*repmat(exp(-LDis(u1,U))',1,size(W,2)) ...
                               .*(repmat(S(s,:),sum(U),1)-W(U,:));
        end
    end

    A = 1:size(PopSel,2);
    U = 1:size(PopSel,2);
    XU = zeros(1,size(PopSel,2));
    for ii = 1:size(PopSel,2)
        x = randi(length(A));
        [~,u] = min(pdist2(PopSel(A(x)).dec, W(U,:)));
        XU(U(u)) = A(x);
        A(x) = [];
        U(u) = [];
    end

    Winner = [];
    if isempty(B) || size(B,2) < 1
        Winner = 1:numel(PopSel);
    else
        for u = 1:size(PopSel,2)
            max_nu = size(B,2);
            randomIndex = randi([1,max_nu]);
            randomSolution = XU(B(u,randomIndex));
            if FitSel(u) > FitSel(randomSolution)
                Winner = [Winner, u]; %#ok<AGROW>
            else
                Winner = [Winner, randomSolution]; %#ok<AGROW>
            end
        end
    end

    if isempty(Winner)
        MatingPool = randi(Npop, 1, Npop);
        return;
    end

    WinnerIdx = idx(Winner);
    MatingPool = WinnerIdx(randi(numel(WinnerIdx),1,Npop));
end
