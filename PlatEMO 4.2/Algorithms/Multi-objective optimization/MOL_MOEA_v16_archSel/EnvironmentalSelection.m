function Population = EnvironmentalSelection(Population,N)
% SPEA2-style environmental selection (v12, 借鉴 FRCSO) —— 向量化加速版。
%   返回恰好 N 个个体: 先按 SPEA2 fitness=R(被支配度)+D(密度) 取 fitness<1 的非支配集;
%   不够 N -> 用 fitness 排名补满(保留被支配点, 关键: 让 sim-rich 的被支配膝点分子留下);
%   超过 N -> SPEA2 truncation 按拥挤度删最密的(维持前沿分布)。
%   统一用于 LSMOP 与分子任务, 不按任务切换。
%   v12 提速: CalFitness 的支配矩阵改用 3D 广播向量化(去掉 O(N^2) 双重 for 循环)。

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. Based on PlatEMO SPEA2.
%--------------------------------------------------------------------------

    if isempty(Population)
        return;
    end

    t1 = tic;
    Fitness = CalFitness(Population.objs);
    tF = toc(t1);

    tT = 0;
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next = false(size(Fitness));
        Next(Rank(1:min(N,length(Rank)))) = true;
    elseif sum(Next) > N
        t2 = tic;
        Del = Truncation(Population(Next).objs,sum(Next)-N);
        tT = toc(t2);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end

    if rand < 0.05   % [ENVT] 偶发计时, 定位耗时 (5% 的代打印一次, 不刷屏)
        fprintf('[ENVT] N=%d nd=%d CalFit=%.0fms Trunc=%.0fms\n', ...
            numel(Population), sum(Fitness<1), tF*1000, tT*1000);
    end

    Population = Population(Next);
end

function Fitness = CalFitness(PopObj)
% SPEA2 fitness = R(raw, 被支配强度和) + D(密度, 第k近邻)。
% 向量化: 用 3D 广播一次算出支配矩阵, 避免 O(N^2) 双重 for。
    N = size(PopObj,1);
    if N == 1
        Fitness = 0; return;
    end

    A = reshape(PopObj, N, 1, []);    % N×1×M
    B = reshape(PopObj, 1, N, []);    % 1×N×M
    Dominate = all(A <= B, 3) & any(A < B, 3);   % Dominate(i,j): i 支配 j
    Dominate(1:N+1:end) = false;                 % 去自身

    S = sum(Dominate, 2);             % 强度: i 支配了多少个 (N×1)
    R = (S' * Dominate)';             % R(i)=sum_j S(j)*Dominate(j,i) = 支配 i 的那些 j 的强度和

    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(N))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);

    Fitness = R + D;                  % 均为 N×1
    Fitness = Fitness(:)';
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
