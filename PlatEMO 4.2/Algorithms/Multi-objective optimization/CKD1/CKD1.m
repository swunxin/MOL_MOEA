classdef CKD1 < ALGORITHM
    methods
        function main(Algorithm,Problem)
            %% ====== 导出历史（原有） ======

            %% ====== 初始化 SOM / CSO ======
            Population = Problem.Initialization();
RealN = length(Population);

% 2) 计算默认网格 D（保证 d>=1）
d = max(1, floor(RealN^(1/(Problem.M-1))));
D_default = repmat(d, 1, Problem.M-1);

% 3) 读参数
[D,tau0,H] = Algorithm.ParameterSet(D_default, 0.7, 5);

% 4) 强制把 D 调到 <= RealN（避免网格大于种群）
D = max(1, floor(D));          % 保证整数且>=1
while prod(D) > RealN
    D = max(1, floor(D * 0.9)); % 缩小一点点直到不超
end
GridN = prod(D);

% 5) 截断/抽样，让 Population 与网格一一对应
if GridN < RealN
    idx = randperm(RealN, GridN);
    Population = Population(idx);
end

Problem.N = GridN;

% 6) 参考向量（如果算法需要）
[V,~] = UniformPoint(Problem.N, Problem.M);

sigma0 = sqrt(sum(D.^2)/(Problem.M-1))/2;

            S       = Population.decs;
            W       = S;
            D = arrayfun(@(S)1:S,D,'UniformOutput',false);
            eval(sprintf('[%s]=ndgrid(D{:});',sprintf('c%d,',1:length(D))))
            eval(sprintf('Z=[%s];',sprintf('c%d(:),',1:length(D))))
            LDis    = pdist2(Z,Z);
            [~,B]   = sort(LDis,2);
            B       = B(:,2:min(H+1,end));

            %% ====== 状态构造器 & 网络 ======
            cfg.numActions     = 2;
            cfg.use_cache_only = true;       % 只吃你 push 的 NDR，不重复计算
            cfg.zero_on_miss   = true;       % 首步/缺缓存 -> 置零

            % ===== 目标方向（关键）：+1=最小化，-1=最大化（会乘-1转成最小化）=====
            % 例：第1目标max，第2目标min
            
            % 8 维特征：1 NDR, 4 pi_prev, 5 PF1_ratio, 6 s4, 7 s5, 8 s6
            cfg.feature_mask   = [1 0 0 1 1 1 1];   % 7位
            cfg.minimize       = false;       % 若为“最大化”Fitness，则置 false

            State('reset');
            prev_pi = 0;
            State('push_cache', struct('have',true,'ndr_prev2cur',0,'c_ba',0,'c_ab',0));  % 首步缓存

            % ===== 首步：先算 Fitness，再构造状态 =====
            pop0     = Population;
            Fitness0 = calFitness(pop0.objs);      % SDE Fitness
            cfg.current_fitness = Fitness0;        % 传入 State，用于 s4/s5/s6

            s0_raw   = State(pop0, prev_pi, cfg);
            w0       = feature_weights(0, cfg.feature_mask);
            s0       = s0_raw .* w0;
            inputSize = numel(s0);

            % === 关键调参 ===
            trainingGap   = 10;
            learnRate     = 0.01;
            temperature   = 0.7;
            entropyBonus  = 0.05;          % 可视需要做退火
            probClip      = [0.1, 0.9];    % [] 代表不裁剪

            outputSize = 2;
            hidden1    = 16;

            layers = [
                sequenceInputLayer(inputSize, "Name","input", "Normalization","none")
                lstmLayer(hidden1, "OutputMode","sequence", "Name","lstm")
                fullyConnectedLayer(outputSize, "Name","fc_out")
            ];
            lgraph = layerGraph(layers);
            dlnet  = dlnetwork(lgraph);

            %% ====== A2C 变量 ======
            V_old_A = 0;  V_old_B = 0;
            RewardBuffer_A = []; RewardBuffer_B = [];

            rewardA_buffer = []; rewardB_buffer = [];
            X_buffer = []; actions_buffer = []; rewards_buffer = [];
            advantages_buffer_A = []; advantages_buffer_B = [];

            trainingGapCounter = 0;

            %% ====== 主循环 ======
            while Algorithm.NotTerminated(Population)

                % ---------- 本代 Fitness（SDE）先算出来，供 State 与决策共用 ----------
                Fitness = calFitness(Population.objs);
                cfg.current_fitness = Fitness;

                % ---------- 状态输入 & 策略采样 ----------
                progress  = min(max(Problem.FE/Problem.maxFE,0),1);  % 0→1
                state_raw = State(Population, prev_pi, cfg);         % 含 s4/s5/s6
                w_t       = feature_weights(progress, cfg.feature_mask);
                state_t   = state_raw .* w_t;
                X_buffer  = [X_buffer, state_t];

                % 记录“本步所选动作的概率”
                [action, pi_vec] = predictLSTMAction(dlnet, X_buffer, temperature, probClip);

                Population_temp = Population;   % 备份

                % ---------- 根据动作执行 SDE / SOM ----------
                if action == 1
                    % === TournamentSelection (ANSGAIII-style) ===
                    Npop = length(Population);
                    if Npop >= 2
                        fit_cons = sum(max(0,Population.cons),2);
                        if all(fit_cons == fit_cons(1))
                            Rank = randperm(Npop, floor(Npop/2)*2);
                        else
                            Rank = TournamentSelection(2, floor(Npop/2)*2, fit_cons);
                        end
                    else
                        Rank = [1,1];
                    end
                    Loser  = Rank(1:end/2);
                    Winner = Rank(end/2+1:end);
                else
                    % === SOM ===
                    Winner = []; Loser = [];
                    for s = 1:size(S,1)
                        sigma  = sigma0*(1-(Problem.FE+s)/Problem.maxFE);
                        tau    = tau0*(1-(Problem.FE+s)/Problem.maxFE);
                        [~,u1] = min(pdist2(S(s,:),W));
                        U      = LDis(u1,:) < sigma;
                        W(U,:) = W(U,:) + tau.*repmat(exp(-LDis(u1,U))',1,size(W,2)) ...
                                           .*(repmat(S(s,:),sum(U),1)-W(U,:));
                    end
                    A = 1:size(Population,2); U = 1:size(Population,2);
                    XU = zeros(1,size(Population,2));
                    for ii = 1:size(Population,2)
                        x = randi(length(A));
                        [~,u] = min(pdist2(Population(A(x)).dec, W(U,:)));
                        XU(U(u)) = A(x); A(x)=[]; U(u)=[];
                    end
                    for u = 1:size(Population,2)
                        max_nu = size(B,2);
                        randomIndex = randi([1,max_nu]);
                        randomSolution = XU(B(u,randomIndex));
                        if Fitness(u) > Fitness(randomSolution)
                            Winner = [Winner, u];
                            Loser  = [Loser, randomSolution];
                        else
                            Winner = [Winner, randomSolution];
                            Loser  = [Loser, u];
                        end
                    end
                end

                % ---------- 生成子代 + 环境选择 ----------
                Offspring  = Operator(Problem,Population(Loser),Population(Winner));
                Population = EnvironmentalSelection([Population,Offspring],V,(Problem.FE/Problem.maxFE)^2);
                Population = Pop_Pool(Population, Population_temp, Offspring, Problem);

                % ---------- 奖励（ComparePop） ----------
                [reward, C_BA, C_AB] = ComparePop(Population_temp, Population);  % reward=ndr
                State('push_cache', struct('have',true,'ndr_prev2cur',reward,'c_ba',C_BA,'c_ab',C_AB));

                prev_pi = pi_vec(action);

                % ---------- 按动作更新 buffer / 优势（即时奖励；非当前动作置 NaN） ----------
                if action == 1
                    rewardA_buffer = [rewardA_buffer, reward];
                    rewardB_buffer = [rewardB_buffer, NaN];

                    r_a = reward;
                    if ~isempty(RewardBuffer_A)
                        rt_mean_A = mean(RewardBuffer_A,'omitnan');
                    else
                        rt_mean_A = r_a;
                    end
                    [V_new_A, A_A] = a_update_critic(r_a, rt_mean_A, V_old_A, trainingGapCounter);
                    V_old_A = V_new_A;

                    advantages_buffer_A = [advantages_buffer_A, A_A];
                    advantages_buffer_B = [advantages_buffer_B, NaN];

                else
                    rewardA_buffer = [rewardA_buffer, NaN];
                    rewardB_buffer = [rewardB_buffer, reward];

                    r_b = reward;
                    if ~isempty(RewardBuffer_B)
                        rt_mean_B = mean(RewardBuffer_B,'omitnan');
                    else
                        rt_mean_B = r_b;
                    end
                    [V_new_B, A_B] = a_update_critic(r_b, rt_mean_B, V_old_B, trainingGapCounter);
                    V_old_B = V_new_B;

                    advantages_buffer_A = [advantages_buffer_A, NaN];
                    advantages_buffer_B = [advantages_buffer_B, A_B];
                end

                actions_buffer = [actions_buffer, action];
                rewards_buffer = [rewards_buffer, reward];

                trainingGapCounter = trainingGapCounter + 1;

                % ---------- 每 trainingGap 步触发 A2C 更新 ----------
                if mod(trainingGapCounter, trainingGap) == 0
                    % 1) 滚动基线
                    if any(~isnan(rewardA_buffer))
                        r_a_win = mean(rewardA_buffer,'omitnan');
                        if numel(RewardBuffer_A) >= 5
                            RewardBuffer_A = [RewardBuffer_A(2:end), r_a_win];
                        else
                            RewardBuffer_A = [RewardBuffer_A, r_a_win];
                        end
                    end
                    if any(~isnan(rewardB_buffer))
                        r_b_win = mean(rewardB_buffer,'omitnan');
                        if numel(RewardBuffer_B) >= 5
                            RewardBuffer_B = [RewardBuffer_B(2:end), r_b_win];
                        else
                            RewardBuffer_B = [RewardBuffer_B, r_b_win];
                        end
                    end

                    % 2) 用原始优势：仅取“该步动作”的优势；其余步填 0
                    T = numel(actions_buffer);
                    advantages_buffer = zeros(1, T, 'like', advantages_buffer_A);

                    idxA = (actions_buffer == 1) & isfinite(advantages_buffer_A);
                    idxB = (actions_buffer == 2) & isfinite(advantages_buffer_B);

                    advantages_buffer(idxA) = advantages_buffer_A(idxA);
                    advantages_buffer(idxB) = advantages_buffer_B(idxB);

                    [dlnet, avgLoss] = updateLSTMActor(dlnet, X_buffer, actions_buffer, ...
                        advantages_buffer, trainingGap, learnRate, temperature, entropyBonus);

                    % 4) 清空窗口缓冲
                    X_buffer = [];
                    actions_buffer = [];
                    rewards_buffer = [];
                    rewardA_buffer = [];
                    rewardB_buffer = [];
                    advantages_buffer_A = [];
                    advantages_buffer_B = [];
                end

                %（保留原有全局历史的赋值）

            end
        end
    end
end

%% ========================= 附属函数 =========================
function Fitness = calFitness(PopObj)
    % SDE Fitness（含归一化）
    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end

function w = feature_weights(~, mask)
    w = ones(sum(mask), 1);
% 元素级权重：长度 = sum(mask)，返回列向量
% 设计：早期更看多样性；后期更看前沿质量与收敛
    % p = max(0,min(1,p));
    % idx = find(mask);
    % w   = ones(numel(idx),1);
    % 
    % function setw(pos, val)
    %     k = find(idx==pos, 1);
    %     if ~isempty(k), w(k) = val; end
    % end
    % 
    % % 1 NDR / 4 pi_prev：恒 1
    % setw(1, 1.0);
    % setw(4, 1.0);
    % 
    % % 5 PF1_ratio：随进度 ↑
    % setw(5, 0.5 + 0.5*p);
    % 
    % % 6 s4（历史最优差距）：随进度 ↑
    % setw(6, 0.5 + 0.5*p);
    % 
    % % 7 s5（全局多样性）：随进度 ↓（早期更强调探索/多样性）
    % setw(7, 1.0 - 0.5*p);

end
