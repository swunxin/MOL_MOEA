classdef FRCSO_N100 < ALGORITHM
% <multi/many> <real> <large/none> <constrained/none>
%
% FRCSO_N100: FRCSO with N/maxFE auto-override DISABLED.
%
%   Original FRCSO doubles N (M=2 / N=100 → N=200) and bumps maxFE
%   (10000 → 15000*D) when those PlatEMO defaults are detected. This
%   variant respects user-supplied Problem.N and Problem.maxFE as-is,
%   for apples-to-apples server batch comparison against algorithms
%   that don't auto-resize.
%
%------------------------------- Reference --------------------------------
% X. Gao, S. Song, H. Zhang, and Z. Wang, A Flexible Ranking-Based
% Competitive Swarm Optimizer for Large-Scale Continuous Multi-Objective
% Optimization, IEEE Transactions on Evolutionary Computation, 2025.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            alpha = Algorithm.ParameterSet(3);

            % NOTE: FRCSO's paper-aligned N/maxFE auto-overrides DISABLED.
            % User-supplied Problem.N and Problem.maxFE are respected as-is.

            %% Generate random population
            Population = Problem.Initialization(Problem.N);
            Population = EnvironmentalSelection(Population,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                PopObj = Population.objs;
                N      = length(Population);
                halfN = floor(N/2);

                %% Step 1: FlexibleRanking
                [WinSet1,LosSet,rankInfo] = FlexibleRanking(PopObj,alpha);

                % Stop condition guard: WinSet1 and LosSet must partition 1:N by N/2 each
                if length(WinSet1) ~= halfN
                    error('FRCSO:StopCondition6','TruncateSelection does not return exactly N/2 winners.');
                end
                if length(LosSet) ~= halfN
                    error('FRCSO:StopCondition2','LosSet does not return exactly N/2 losers.');
                end
                if length(unique([WinSet1(:);LosSet(:)])) ~= N
                    error('FRCSO:StopCondition2','WinSet1 U LosSet does not cover 1:N exactly once.');
                end

                %% Step 2: LoserUpdating (staged by |F1|)
                if rankInfo.lenF1 < halfN
                    % early stage: LMOCSO-style single-winner update
                    W1 = SampleWinnersForLosers(WinSet1,halfN);
                    [WinSet2Dec,WinSet2Vel] = OperatorEarly(Problem,Population(LosSet),Population(W1));
                else
                    % late stage: two-winner update (deterministic cyclic pairing)
                    [W1,W2] = TwoWinnerPairing(WinSet1,halfN);
                    if any(W1 == W2)
                        error('FRCSO:StopCondition7','Two-winner pairing must satisfy xw1 ~= xw2.');
                    end
                    [WinSet2Dec,WinSet2Vel] = OperatorLate(Problem,Population(LosSet),Population(W1),Population(W2));
                end

                % Stop condition guard: updated losers must be N/2
                if size(WinSet2Dec,1) ~= halfN
                    error('FRCSO:StopCondition3','LoserUpdating output size is not N/2.');
                end

                %% Step 3: Offspring = Mutation(WinSet1 U WinSet2)
                WinnerDec = Population(WinSet1).decs;
                WinnerVel = Population(WinSet1).adds(zeros(halfN,size(WinnerDec,2)));
                OffDec = [WinnerDec;WinSet2Dec];
                OffVel = [WinnerVel;WinSet2Vel];
                Offspring = MutationEvaluate(Problem,OffDec,OffVel);

                if length(Offspring) ~= N
                    error('FRCSO:StopCondition3','Offspring size is not N.');
                end

                %% Step 3: P = EnvironmentalSelection(P U Offspring)
                Population = EnvironmentalSelection([Population,Offspring],N);
            end
        end
    end
end

%% ============ HELPER FUNCTIONS ============

function Dis = CrowdingDistance(PopObj)
    [N,M] = size(PopObj);
    Dis = zeros(N,1);
    if N == 0
        return;
    elseif N <= 2
        Dis(:) = inf;
        return;
    end
    for m = 1:M
        [~,idx] = sort(PopObj(:,m));
        Dis(idx(1)) = Inf;
        Dis(idx(end)) = Inf;
        for i = 2:N-1
            Dis(idx(i)) = Dis(idx(i)) + (PopObj(idx(i+1),m) - PopObj(idx(i-1),m)) ...
                / (max(PopObj(:,m)) - min(PopObj(:,m)) + eps);
        end
    end
end

function LD = LocalDegreeCalculation(ArcSetObj,fmax,fmin,delta)
    [N,M] = size(ArcSetObj);
    LD = zeros(N,1);
    R = delta .* (fmax - fmin);
    for i = 1:N
        xi = ArcSetObj(i,:);
        inN = true(N,1);
        for j = 1:M
            inN = inN & (ArcSetObj(:,j) >= xi(j)-R(j)) & (ArcSetObj(:,j) <= xi(j)+R(j));
        end
        Ci = find(inN & ((1:N)' ~= i));
        if isempty(Ci)
            LD(i) = 0;
            continue;
        end
        B = 0;
        for j = Ci'
            xj = ArcSetObj(j,:);
            if all(xj <= xi) && any(xj < xi)
                B = B + 1;
            end
        end
        LD(i) = B / length(Ci);
    end
end

function ArcSort = PacketSequencing(ArcSetIdx,LocalDegree,DisDegree)
% Approximation: lexicographic packet sequencing by (LocalDegree asc, CrowdingDistance desc, index asc)
% Keep "smaller local degree is better", and use larger crowding distance for tie-break.
    [~,ArcSortIdx] = sortrows([LocalDegree(:),-DisDegree(:),ArcSetIdx(:)]);
    ArcSort = ArcSetIdx(ArcSortIdx);
end

function [WinSet1,LosSet,info] = FlexibleRanking(PopObj,alpha)
    N = size(PopObj,1);
    [FrontNo,~] = NDSort(PopObj,1);
    F1 = find(FrontNo == 1)';
    lenF1 = length(F1);
    info.lenF1 = lenF1;

    if lenF1 < N
        ArcSetIdx = setdiff(1:N,F1,'stable');
        ArcSetObj = PopObj(ArcSetIdx,:);
        DisDegree = CrowdingDistance(ArcSetObj);
        fmax = max(PopObj,[],1);
        fmin = min(PopObj,[],1);
        delta = alpha^((lenF1/N) - 1);
        LocalDegree = LocalDegreeCalculation(ArcSetObj,fmax,fmin,delta);
        ArcSort = PacketSequencing(ArcSetIdx,LocalDegree,DisDegree);
    else
        ArcSort = [];
    end

    WinSet1 = TruncateSelection(F1,ArcSort,N);
    LosSet  = setdiff(1:N,WinSet1,'stable');
    halfN   = floor(N/2);
    if length(LosSet) > halfN
        LosSet = LosSet(1:halfN);
    elseif length(LosSet) < halfN
        FillPool = setdiff(1:N,[WinSet1,LosSet],'stable');
        LosSet = [LosSet,FillPool(1:(halfN-length(LosSet)))];
    end
end

function WinSet1 = TruncateSelection(F1,ArcSort,N)
    lenF1 = length(F1);
    halfN = floor(N/2);
    qN    = floor(N/4);
    F1    = F1(:)';
    ArcSort = ArcSort(:)';

    if lenF1 < halfN
        need = halfN - lenF1;
        WinSet1 = [F1,ArcSort(1:min(need,length(ArcSort)))];
    elseif lenF1 < 3*N/4
        takeF1 = min(qN,length(F1));
        takeArc = min(qN,length(ArcSort));
        if takeF1 > 0
            selectF1 = RandomSelect(F1,takeF1);
        else
            selectF1 = [];
        end
        selectArc = ArcSort(1:takeArc);
        WinSet1 = [selectF1(:)',selectArc(:)'];
    else
        WinSet1 = RandomSelect(F1,halfN);
    end

    WinSet1 = unique(WinSet1,'stable');
    if length(WinSet1) < halfN
        FillPool = [setdiff(F1,WinSet1,'stable'),setdiff(ArcSort,WinSet1,'stable')];
        if length(FillPool) < halfN-length(WinSet1)
            FillPool = [FillPool,setdiff(1:N,[WinSet1,FillPool],'stable')];
        end
        WinSet1 = [WinSet1,FillPool(1:(halfN-length(WinSet1)))];
    end
    WinSet1 = WinSet1(1:halfN);
end

function W1 = SampleWinnersForLosers(WinSet1,nLosers)
    pool = WinSet1(:)';
    if isempty(pool)
        W1 = [];
        return;
    end
    if length(pool) >= nLosers
        W1 = pool(randperm(length(pool),nLosers));
    else
        rep = repmat(pool,1,ceil(nLosers/length(pool)));
        W1  = rep(1:nLosers);
    end
end

function [W1,W2] = TwoWinnerPairing(WinSet1,nLosers)
% Deterministic approximation: cyclic two-winner pairing from the current winner pool.
    pool = WinSet1(:)';
    if length(pool) < 2
        W1 = repmat(pool,1,nLosers);
        W2 = W1;
        return;
    end
    rep1 = repmat(pool,1,ceil(nLosers/length(pool)));
    rep2 = repmat([pool(2:end),pool(1)],1,ceil(nLosers/length(pool)));
    W1 = rep1(1:nLosers);
    W2 = rep2(1:nLosers);
end

function sample = RandomSelect(pool,k)
    if k <= 0
        sample = [];
        return;
    end
    idx = randperm(length(pool),k);
    sample = pool(idx);
end

function Offspring = MutationEvaluate(Problem,OffDec,OffVel)
    [N,D] = size(OffDec);
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    disM  = 20;

    Site = rand(N,D) < 1/D;
    mu   = rand(N,D);
    OffDec = max(min(OffDec,Upper),Lower);

    temp = Site & mu<=0.5;
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

    Offspring = Problem.Evaluation(OffDec,OffVel);
end
