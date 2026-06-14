classdef A2C_IM_OPTWO < ALGORITHM
    methods
        function main(Algorithm,Problem)
            %% ====== 参数初始化 ======
            % SOM 参数（D 每维神经元数，tau0 学习率，H 邻域交配池大小）
            [D,tau0,H] = Algorithm.ParameterSet(repmat(ceil(Problem.N.^(1/(Problem.M-1))),1,Problem.M-1),0.7,5);
            % 均匀参考点 + 维护N
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Problem.N      = prod(D);

            % 初始种群
            Population     = Problem.Initialization();

            % SOM 网格与距离
            sigma0 = sqrt(sum(D.^2)/(Problem.M-1))/2;
            S      = Population.decs;     % 初始训练集
            W      = S;                   % 神经元权重
            D      = arrayfun(@(S)1:S,D,'UniformOutput',false);
            eval(sprintf('[%s]=ndgrid(D{:});',sprintf('c%d,',1:length(D))))
            eval(sprintf('Z=[%s];',sprintf('c%d(:),',1:length(D))))
            LDis   = pdist2(Z,Z);
            [~,B]  = sort(LDis,2);
            B      = B(:,2:min(H+1,end));

            %% ====== A2C 与计数器 ======
            w_A = 0.5;  w_B = 0.5;
            V_old_A = 0; V_old_B = 0;
            RewardBuffer_A = []; RewardBuffer_B = [];
            trainingGapCounter = 0;

            %% ====== 主循环 ======
            while Algorithm.NotTerminated(Population)

                %------------- 计算权重并选择策略 -------------
                weight_combine = w_combine(w_A, w_B);

                f_a = 0; f_b = 0;                     % 策略标识
                Population_temp = Population;         % A：旧代（用于奖励比较）
                Fitness = calFitness(Population.objs);

                randomPro = rand();
               if randomPro < weight_combine
                    %% SDE
                    % LMOCSO的环境选择
                    f_a = 1;
                    %Population    = EnvironmentalSelection(Population,V,(Problem.FE/Problem.maxFE)^2);
                    if length(Population) >= 2
                        % 生成从1到种群大小的随机排列的序列，序列的长度是种群大小的一半（向下取整）的两倍（即一定为偶数）
                        Rank = randperm(length(Population),floor(length(Population)/2)*2);
                    else
                        Rank = [1,1];
                    end
                    Loser  = Rank(1:end/2);
                    Winner = Rank(end/2+1:end);
                    Change = Fitness(Loser) >= Fitness(Winner);
                    Temp   = Winner(Change);
                    Winner(Change) = Loser(Change);
                    Loser(Change)  = Temp;
                else
                    %% SOM
                    f_b = 1;
                    % SOM初始化
                    Winner = [];
                    Loser = [];                  
                    % Update SOM
                    %对于训练集中的每个样本
                    for s = 1 : size(S,1)
                        % 计算邻域半径 sigma
                        sigma  = sigma0*(1-(Problem.FE+s)/Problem.maxFE);
                        % 计算学习率 tau
                        tau    = tau0*(1-(Problem.FE+s)/Problem.maxFE);
                        % 找出权重向量 W 中最接近该样本的神经元 u1
                        [~,u1] = min(pdist2(S(s,:),W));
                        % 找到在当前邻域半径内的所有神经元
                        U      = LDis(u1,:) < sigma;
                        % 更新在邻域内的所有神经元的权重
                        W(U,:) = W(U,:) + tau.*repmat(exp(-LDis(u1,U))',1,size(W,2)).*(repmat(S(s,:),sum(U),1)-W(U,:));
                    end
                    
                    % Associate each solution with a neuron 将每个解与一个神经元关联
                    % 生成一个从 1 到 Problem.N 的序列 A
                    A  = 1 : size(Population, 2);
                    U  = 1 : size(Population, 2);
                    % 初始化一个长度为N的零向量  用于存储与神经元u最近的解决方案的索引
                    XU = zeros(1,size(Population, 2));
                    % 对于每个解
                    for i = 1 : size(Population, 2)
                        % 随机选择一个解决方案
                        x        = randi(length(A));
                        % 找出权重向量 W 中最接近该元素的神经元 u
                        % pdist2(Population(A(x)).dec,W(U,:))计算了当前解决方案与每个未被匹配的神经元之间的欧氏距离
                        [~,u]    = min(pdist2(Population(A(x)).dec,W(U,:)));
                        % 将 u 与该元素关联
                        XU(U(u)) = A(x);
                        % 从解决方案集中移除该解决方案
                        A(x)     = [];
                        % 从神经元集合中移除已找到的神经元
                        U(u)     = [];
                    end
                    for u = 1 : size(Population, 2)
                        % 从 XU(B(u,:)) 中随机选择一个解
                        max_nu = size(B, 2);
                        randomIndex = randi([1,max_nu]);
                        randomSolution = XU(B(u,randomIndex));
                        % 将当前解和随机选择的邻域解进行 Fitness 的比较
                        if Fitness(u) > Fitness(randomSolution)
                            % 如果当前解的 Fitness 大于随机选择的解，将当前解的序号写入 Winner，将随机选择的解写入 Loser
                            Winner = [Winner, u];
                            Loser  = [Loser, randomSolution];
                        else
                            % 如果当前解的 Fitness 小于或等于随机选择的解，将当前解的序号写入 Loser，将随机选择的解写入 Winner
                            Winner = [Winner, randomSolution];
                            Loser  = [Loser, u];
                        end
                    end
                    numPairs = numel(Loser);
                    allIdx   = 1 : numel(Population);
                    PeerIdx  = zeros(1,numPairs);
                    
                    for k = 1:numPairs
                        % 从种群随机挑一个同伴，避免挑到本对里的 winner/loser 本人
                        pool = setdiff(allIdx, [Winner(k), Loser(k)]);
                        PeerIdx(k) = pool(randi(numel(pool)));
                    end
                end % 策略选择终止

                % 生成子代（B_raw）
                tag = f_b;
                if tag
                   Offspring = Operator_SOM(Problem, Population(Loser), Population(Winner), Population(PeerIdx));
                else 

                    Offspring      = Operator_SDE(Problem,Population(Loser),Population(Winner));
                end
                %Offspring = Operator(Problem,Population(Loser),Population(Winner));
                Population = EnvironmentalSelection([Population, Offspring], V, (Problem.FE/Problem.maxFE)^2);
                Population = Pop_Pool(Population, Population_temp, Offspring, Problem); % Step3只从Offspring前M个新点补齐


             %B_tilde = EnvironmentalSelection(Offspring, V, (Problem.FE/Problem.maxFE)^2);
                %------------- Training Gap 计数（先+1，马上可能用于奖励更新） -------------
                trainingGapCounter = trainingGapCounter + 1;

                %------------- 每 10 次做一次 NDR 奖励（A vs 子代） -------------
                if mod(trainingGapCounter,10) == 0
                    if f_a == 1
                        % ★ 奖励在环境选择之前：A（旧代） vs B（子代）
                        [r_a,C_BA,C_AB] = ComparePop(Population_temp, Population);

                         if length(RewardBuffer_A) >= 3
                            RewardBuffer_A = [RewardBuffer_A(2:end), r_a];  % 更新奖励缓冲区
                        else
                            RewardBuffer_A = [RewardBuffer_A, r_a];
                        end
                        rt_mean_A = mean(RewardBuffer_A);  % 计算奖励均值
                        [V_new_A, A_A] = a_update_critic(r_a, rt_mean_A, V_old_A, trainingGapCounter);  % 更新状态值
                        w_A = w_update_actor(A_A, w_A);  % 更新权重
                        V_old_A = V_new_A;  % 更新状态值
                        r_temp=r_a; % 保存奖励值，方便存储
                  
                   elseif f_b == 1

                        % ★ 奖励在环境选择之前：A（旧代） vs B（子代）
                        [r_b,C_BA,C_AB] = ComparePop(Population_temp, Population);

                        if length(RewardBuffer_B) >= 3
                            RewardBuffer_B = [RewardBuffer_B(2:end), r_b];  % 更新奖励缓冲区
                        else
                            RewardBuffer_B = [RewardBuffer_B, r_b];
                        end
                        rt_mean_B = mean(RewardBuffer_B);  % 计算奖励均值
                        [V_new_B, A_B] = a_update_critic(r_b, rt_mean_B, V_old_B, trainingGapCounter);  % 更新状态值
                        w_B = w_update_actor(A_B, w_B);  % 更新权重
                        V_old_B = V_new_B;  % 更新状态值
                        r_temp=r_b; % 保存奖励值，方便存储
                    end
                end
                
                %------------- 环境选择（精英） -------------
              
            end
        end
    end
end

%% ====== 工具函数 ======
function Fitness = calFitness(PopObj)
% shift-based density（加eps防除零更稳健）
    N    = size(PopObj,1);
    fmax = max(PopObj,[],1);
    fmin = min(PopObj,[],1);
    span = max(fmax-fmin, eps);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(span,N,1);

    Dis  = inf(N);
    for i = 1:N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end

