function Population = EnvironmentalSelection(Population,N)
% SPEA2-style environmental selection (v12, 借鉴 FRCSO)。
%   返回恰好 N 个个体: 先按 SPEA2 fitness=R(被支配度)+D(密度) 取 fitness<1 的非支配集;
%   不够 N -> 用 fitness 排名补满(保留被支配点, 关键: 让 sim-rich 的被支配膝点分子留下);
%   超过 N -> SPEA2 truncation 按拥挤度删最密的(维持前沿分布)。
%   统一用于 LSMOP 与分子任务, 不按任务切换。

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. Based on PlatEMO SPEA2. Please reference
% "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB
% platform for evolutionary multi-objective optimization, IEEE CIM, 2017".
%--------------------------------------------------------------------------

    if isempty(Population)
        return;
    end

    Fitness = CalFitness(Population.objs);

    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next = false(size(Fitness));
        Next(Rank(1:min(N,length(Rank)))) = true;
    elseif sum(Next) > N
        Del = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end

    Population = Population(Next);
end

function Fitness = CalFitness(PopObj)
    N = size(PopObj,1);

    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end

    S = sum(Dominate,2);
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end

    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);

    Fitness = R + D';
end

function Del = Truncation(PopObj,K)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain = find(~Del);
        Temp = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
