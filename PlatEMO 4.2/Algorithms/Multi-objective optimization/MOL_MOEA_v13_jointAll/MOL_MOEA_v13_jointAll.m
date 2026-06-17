classdef MOL_MOEA_v13_jointAll < ALGORITHM
% MOL_MOEA_v13_jointAll: == MOL_MOEA_v12_envSPEA2 的进化(SPEA2环境选择)一字不动,
%   把 v13 的"联合(目标⊕潜)空间 BANK + 输出合并"策略 **无分子开关** 地施加到 LSMOP。
%
%   背景: v12 里 BANK/merge 对 LSMOP 是 gated-off 死代码(merge 要 ~isempty(objThr));
%   本版去掉该门, 让双空间档案真正进入 LSMOP 被测输出, 作为"v12替换环境选择后, 进一步的
%   高维多目标演化改进"探针。隔离 merge 效应: 进化(EnvironmentalSelection 目标SPEA2)与v12逐字相同,
%   只有 报告种群 = jointSelect([pop ∪ BANK], N) 变了。
%
%   - updateBank: front-1 近前沿档案, 用联合(目标⊕潜)密度维护(取代 v12 潜 niche/migrate)。
%   - mergeForOutput -> jointSelect: 在 [pop ∪ BANK] 上做"双空间 SPEA2":
%       R = 目标空间支配强度(收敛, 同 SPEA2) + D = 联合(目标⊕潜)密度(非支配集按目标+决策谱铺),
%       截断也用联合密度。被支配点高 R 不入选 => IGD 安全(只能找回 pop 丢掉的决策多样近前沿点)。
%   - 联合距离: 目标块/潜块各 min-max 归一化后按维度数取均方平衡(防高维潜淹没低维目标), 开方相加。
%   - 无阈值/无 niche 半径; r_frac/K_max 弃用(保留签名兼容驱动)。注入(L2)仍默认关。
%
%   [以下为 v12 原说明]
% MOL_MOEA_v12_envSPEA2: == MOL_MOEA_v10_bank, 唯一改动 = 种群维护机制。
%   把 v10 的"两段式" [ EnvironmentalSelection(RVEA/参考向量+APD) + Pop_Pool(取F1+池补满) ]
%   换成 FRCSO 的"一段式" SPEA2 环境选择(R被支配度+密度D, 不够N用被支配点补满, 超N按拥挤度truncate),
%   直接返回 N。动机(2026-06 诊断): 我们与FRCSO在LSMOP持平、分子大幅落后, 差距=qed-sim前沿膝点分布;
%   根因="只留第一前沿"在 env-selection 和 Pop_Pool 各有一份 -> 分子不规则前沿塌成6-15点、膝点空、
%   sim-rich(被支配)分子被硬丢。SPEA2式保留被支配点+密度分布 -> 膝点覆盖。统一一套选择(LSMOP/分子同用),
%   不是按任务切换。控制器/算子/reward/POP_BANK 全部不动。
%
%   [原 v10_bank 头部如下]
% MOL_MOEA_v10_bank: v7 controller (UNCHANGED) + POP_BANK (dual-space lead archive).
%
%   Built ON v7 (NOT v9). The v7 evolution (controller / env-selection / Pop_Pool)
%   is byte-identical; POP_BANK is added in 3 independently-switchable LAYERS of
%   increasing risk, so v7's LSMOP advantage is never broken before it's proven safe:
%
%     L0 bankOn   : archive ONLY (RNG-free; touches neither evolution nor output).
%                   -> Stage0 (bankOn1,mergeOut0,injectOn0) MUST reproduce v7 exactly.
%     L1 mergeOut : at every gen the REPORTED population (OutputPop) = best-N of
%                   [evolving pop UNION BANK]; the EVOLVING pop stays == v7.
%                   -> output can only recover lost near-front points => IGD <= v7 (safe upside).
%     L2 injectOn : demand-driven + quality-gated injection INTO the evolving pop
%                   (the risky lever). Held OFF by default; staged with small inject_max.
%
%   BANK = dual-gate archive: a candidate is kept iff it is (objective space) NEAR
%   THE FRONT and (decision space) a NEW niche (> radius r from every rep). The
%   r-admission IS the diversity maintenance (reps are pairwise > r apart). Within
%   a niche a Pareto-better candidate MIGRATES (replaces) the rep, so reps track
%   the advancing front. Stored individuals carry objs -> re-injection costs NO FE.
%   Molecular: replace the near-front gate with the property threshold (anchors
%   then accumulate as a lead library) and merge BANK into the final output.
%
%   ParameterSet adds: bankOn(1), mergeOut(1), injectOn(0), inject_max(5),
%                      r_frac(0.1), K_max(30).
%
%   [original v7 header below]
% MOL_MOEA_v7: v5a (winning controller) + DETERMINISTIC LSTM init (fc_out=0).
%
%   ONLY change vs v5a: the output layer is zeroed after construction so every
%   seed starts at pi=[0.5,0.5] (like the bandit's fixed w start). Targets the
%   LSMOP7/9 bimodal variance, whose root is the seed-dependent EARLY policy
%   (gens 1-25, NCR phase) — reward switching (v6) does NOT change the basin.
%   entropyBonus=0, learnRate=0.01, temperature=0.7 (all = v5a).
%
%   [original v5a header below]
% MOL_MOEA_v5a: v3 (SOM-fixed) + LSTM controller knobs (entropyBonus, learnRate).
%
%   Inherits v3 SOM topology. The ONLY new thing vs v3 is exposing two LSTM
%   training hyperparameters as Algorithm.parameter knobs, to test the
%   controller-audit hypothesis (2026-06-03):
%     entropy term (0.05) was ~2.5x larger than the advantage signal (~0.014),
%     so it pushed the policy back to uniform every training step and the
%     LSTM could not keep the early-learned branch lean (unlike the bandit
%     whose w_A/w_B integrate+freeze it). probClip=[0.1,0.9] already floors
%     exploration, so the entropy term is removable.
%
%   Parameters (Algorithm.parameter): D, tau0, H, entropyBonus(=0), learnRate(=0.01)
%     entropyBonus DEFAULT 0  -> entropy term DELETED (test of the hypothesis)
%     learnRate    DEFAULT 0.01 (knob: the live-signal window is only ~25 gens
%                  = ~2-3 updates, so a larger lr may be needed to learn in time)
%   temperature stays fixed at 0.7. reward/pool/SOM unchanged.
    methods
        function main(Algorithm,Problem)
            %% Parameter setting + SMEA-style SOM grid (v3 fix) + LSTM knobs
            [D,tau0,H,entropyBonus,learnRate,bankOn,mergeOut,injectOn,inject_max,r_frac,K_max,task] = Algorithm.ParameterSet( ...
                repmat(ceil(Problem.N.^(1/(Problem.M-1))),1,Problem.M-1), 0.7, 5, 0, 0.01, 1, 1, 0, 5, 0.1, 30, 0);
            % task: MOMO application flag. 0 = NONE (default) -> objThr=[] -> output
            %   merge is dominance-only -> LSMOP/smoke == v7. 1/2/3 = MOMO task
            %   thresholds (objectives are obj = -property, so "meet bar" = obj<=objThr):
            switch task
                case 1;    objThr = [-0.9, -0.4];        % Task1 QED : qed>=0.9 & sim>=0.4
                case 2;    objThr = [ inf, -0.4];        % Task2 pLogP: sim>=0.4 (pLogP improved via dominance)
                case 3;    objThr = [-0.8, -0.3, -0.4];  % Task3 : qed>=0.8 & drd2>=0.3 & sim>=0.4
                otherwise; objThr = [];                  % none -> dominance-only merge
            end
            BANK = [];   % POP_BANK: dual-gate (near-front + decision-niche) lead archive

            [V,Problem.N] = UniformPoint(Problem.N, Problem.M);
            Problem.N     = prod(D);
            sigma0        = sqrt(sum(D.^2)/(Problem.M-1))/2;

            Population = Problem.Initialization();
            RealN      = length(Population);

            % Task2 pLogP bar = the LEAD's pLogP. The molecular Problem (e.g. DDProblem1)
            % builds the initial population around the lead with Population(1)=lead, and
            % objs come from Python as [-pLogP,-sim]; so the lead pLogP = -Population(1).obj(1).
            % Admit a molecule if obj1 <= Population(1).obj(1) (pLogP >= lead) AND sim>=0.4.
            if task == 2 && RealN >= 1
                objThr(1) = Population(1).obj(1);
            end

            S = Population.decs;
            W = S;

            % Neuron lattice Z via ndgrid over 1:D (latent dim = M-1)
            Dcell = arrayfun(@(s) 1:s, D, 'UniformOutput', false);
            eval(sprintf('[%s]=ndgrid(Dcell{:});', sprintf('c%d,',1:numel(D))));
            eval(sprintf('Z=[%s];',                sprintf('c%d(:),',1:numel(D))));
            LDis = pdist2(Z,Z);

            [~,B] = sort(LDis,2);
            if size(B,2) > 1
                B = B(:,2:min(H+1,size(B,2)));
            else
                B = ones(size(B,1),1);
            end

            %% State builder and policy network
            cfg.numActions     = 2;
            cfg.use_cache_only = true;
            cfg.zero_on_miss   = true;
            cfg.feature_mask   = [1 0 0 1 1 1 1];
            cfg.minimize       = false;

            State('reset');
            prev_pi = 0;
            State('push_cache', struct('have',true,'ndr_prev2cur',0,'c_ba',0,'c_ab',0));

            pop0     = Population;
            Fitness0 = calFitness(pop0.objs);
            cfg.current_fitness = Fitness0;

            s0_raw    = State(pop0, prev_pi, cfg);
            w0        = feature_weights(0, cfg.feature_mask);
            s0        = s0_raw .* w0;
            inputSize = numel(s0);

            trainingGap  = 10;
            % learnRate, entropyBonus now come from ParameterSet (knobs).
            temperature  = 0.7;
            probClip     = [0.1, 0.9];
            outputSize   = 2;
            hidden1      = 16;

            layers = [
                sequenceInputLayer(inputSize, "Name","input", "Normalization","none")
                lstmLayer(hidden1, "OutputMode","sequence", "Name","lstm")
                fullyConnectedLayer(outputSize, "Name","fc_out")
            ];
            lgraph = layerGraph(layers);
            dlnet  = dlnetwork(lgraph);
            % v7 DETERMINISTIC INIT: zero the output layer so the initial
            % logits are 0 -> pi = [0.5,0.5] for EVERY seed (like the bandit's
            % fixed w_A=w_B=0.5 start). Random init otherwise gives each seed a
            % different early policy -> different basin -> LSMOP7/9 bimodal variance.
            idxW = strcmp(dlnet.Learnables.Layer,'fc_out') & strcmp(dlnet.Learnables.Parameter,'Weights');
            idxB = strcmp(dlnet.Learnables.Layer,'fc_out') & strcmp(dlnet.Learnables.Parameter,'Bias');
            dlnet.Learnables.Value{idxW} = dlarray(zeros(size(dlnet.Learnables.Value{idxW}),'like',dlnet.Learnables.Value{idxW}));
            dlnet.Learnables.Value{idxB} = dlarray(zeros(size(dlnet.Learnables.Value{idxB}),'like',dlnet.Learnables.Value{idxB}));

            %% A2C variables
            V_old_A = 0;
            V_old_B = 0;
            RewardBuffer_A = [];
            RewardBuffer_B = [];

            rewardA_buffer = [];
            rewardB_buffer = [];
            X_buffer = [];
            actions_buffer = [];
            rewards_buffer = [];
            advantages_buffer_A = [];
            advantages_buffer_B = [];

            trainingGapCounter = 0;

            %% POP_BANK init (B0: anchor initial leads before any selection can drop them)
            if bankOn
                BANK = updateBank(BANK, Population, r_frac, K_max, objThr);
            end
            if bankOn && mergeOut          % v13_jointAll: 去掉 ~isempty(objThr) 门 -> LSMOP 也走 merge
                OutputPop = mergeForOutput(Population, BANK, objThr);   % reported population
            else
                OutputPop = Population;
            end

            %% Main loop  (evolution drives Population==v7; NotTerminated records OutputPop)
            while Algorithm.NotTerminated(OutputPop)
                Fitness = calFitness(Population.objs);
                cfg.current_fitness = Fitness;

                progress  = min(max(Problem.FE/Problem.maxFE,0),1);
                state_raw = State(Population, prev_pi, cfg);
                w_t       = feature_weights(progress, cfg.feature_mask);
                state_t   = state_raw .* w_t;
                X_buffer  = [X_buffer, state_t]; %#ok<AGROW>

                [action, pi_vec] = predictLSTMAction(dlnet, X_buffer, temperature, probClip);

                Population_temp = Population;

                if action == 1
                    if numel(Population) >= 2
                        Rank = randperm(numel(Population), floor(numel(Population)/2)*2);
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
                    % === SOM ===
                    S = Population.decs;   % 每代用当前种群潜变量驱动 SOM
                    Winner = []; Loser = [];
                    for s = 1:size(S,1)
                        sigma = max(0, sigma0*(1-(Problem.FE+s)/Problem.maxFE));
                        tau   = max(0, tau0*(1-(Problem.FE+s)/Problem.maxFE));
                        [~,u1] = min(pdist2(S(s,:),W));
                        U = LDis(u1,:) < sigma;
                        if any(U)
                            W(U,:) = W(U,:) + tau.*repmat(exp(-LDis(u1,U))',1,size(W,2)) ...
                                               .*(repmat(S(s,:),sum(U),1)-W(U,:));
                        end
                    end

                    A  = 1:numel(Population);
                    U  = 1:numel(Population);
                    XU = zeros(1,numel(Population));
                    for ii = 1:numel(Population)
                        x = randi(numel(A));
                        [~,u] = min(pdist2(Population(A(x)).dec, W(U,:)));
                        XU(U(u)) = A(x);
                        A(x) = [];
                        U(u) = [];
                    end

                    % ★ BMU fix（对齐 MOEA_RLD git 版）：用 XU(u) 取实际个体
                    % 原 bug：直接用 neuron index u 当 individual index，导致
                    % Fitness(u) 读的是种群第 u 个的 fitness，而不是 neuron u
                    % 关联的真实个体的 fitness（XU 是排列，identity 不成立）。
                    for u = 1:numel(Population)
                        centerSolution = XU(u);                                  %#ok<NASGU>
                        randomIndex    = randi(size(B,2));
                        randomSolution = XU(B(u,randomIndex));
                        if Fitness(centerSolution) > Fitness(randomSolution)
                            Winner = [Winner, centerSolution]; %#ok<AGROW>
                            Loser  = [Loser, randomSolution];  %#ok<AGROW>
                        else
                            Winner = [Winner, randomSolution]; %#ok<AGROW>
                            Loser  = [Loser, centerSolution];  %#ok<AGROW>
                        end
                    end

                    numPairs = numel(Loser);
                    allIdx   = 1:numel(Population);
                    PeerIdx  = zeros(1,numPairs);
                    for k = 1:numPairs
                        pool = setdiff(allIdx, [Winner(k), Loser(k)]);
                        if isempty(pool)
                            PeerIdx(k) = Winner(k);
                        else
                            PeerIdx(k) = pool(randi(numel(pool)));
                        end
                    end

                    targetPairs = floor(numel(Population)/2);
                    if numel(Loser) > targetPairs
                        keep    = randperm(numel(Loser), targetPairs);
                        Loser   = Loser(keep);
                        Winner  = Winner(keep);
                        PeerIdx = PeerIdx(keep);
                    end
                end

                if action == 1
                    Offspring = Operator_SDE(Problem,Population(Loser),Population(Winner));
                else
                    Offspring = Operator_SOM(Problem,Population(Loser),Population(Winner),Population(PeerIdx));
                end

                %% L0 archive: from the FULL candidate pool, BEFORE env-selection can drop a lead
                if bankOn
                    BANK = updateBank(BANK, [Population_temp, Offspring], r_frac, K_max, objThr);
                end

                % v12: 一段式 SPEA2 环境选择直接返回 N (取代 RVEA选择 + Pop_Pool)。
                Population = EnvironmentalSelection([Population,Offspring], Problem.N);

                %% L2 inject (risky lever, default OFF): refill LOST good directions into evolving pop
                if bankOn && injectOn
                    Population = injectBank(Population, BANK, inject_max);
                end
                %% L1 output merge: report jointSelect([pop ∪ BANK], N); evolution unchanged. v13: 无 objThr 门
                if bankOn && mergeOut
                    OutputPop = mergeForOutput(Population, BANK, objThr);
                else
                    OutputPop = Population;
                end

                [reward, C_BA, C_AB] = ComparePop(Population_temp, Population);
                State('push_cache', struct('have',true,'ndr_prev2cur',reward,'c_ba',C_BA,'c_ab',C_AB));

                prev_pi = pi_vec(action);

                if action == 1
                    rewardA_buffer = [rewardA_buffer, reward]; %#ok<AGROW>
                    rewardB_buffer = [rewardB_buffer, NaN]; %#ok<AGROW>

                    r_a = reward;
                    if ~isempty(RewardBuffer_A)
                        rt_mean_A = mean(RewardBuffer_A,'omitnan');
                    else
                        rt_mean_A = r_a;
                    end
                    [V_new_A, A_A] = a_update_critic(r_a, rt_mean_A, V_old_A, trainingGapCounter);
                    V_old_A = V_new_A;

                    advantages_buffer_A = [advantages_buffer_A, A_A]; %#ok<AGROW>
                    advantages_buffer_B = [advantages_buffer_B, NaN]; %#ok<AGROW>
                else
                    rewardA_buffer = [rewardA_buffer, NaN]; %#ok<AGROW>
                    rewardB_buffer = [rewardB_buffer, reward]; %#ok<AGROW>

                    r_b = reward;
                    if ~isempty(RewardBuffer_B)
                        rt_mean_B = mean(RewardBuffer_B,'omitnan');
                    else
                        rt_mean_B = r_b;
                    end
                    [V_new_B, A_B] = a_update_critic(r_b, rt_mean_B, V_old_B, trainingGapCounter);
                    V_old_B = V_new_B;

                    advantages_buffer_A = [advantages_buffer_A, NaN]; %#ok<AGROW>
                    advantages_buffer_B = [advantages_buffer_B, A_B]; %#ok<AGROW>
                end

                actions_buffer = [actions_buffer, action]; %#ok<AGROW>
                rewards_buffer = [rewards_buffer, reward]; %#ok<AGROW>

                trainingGapCounter = trainingGapCounter + 1;

                if mod(trainingGapCounter, trainingGap) == 0
                    if any(~isnan(rewardA_buffer))
                        r_a_win = mean(rewardA_buffer,'omitnan');
                        if numel(RewardBuffer_A) >= 5
                            RewardBuffer_A = [RewardBuffer_A(2:end), r_a_win];
                        else
                            RewardBuffer_A = [RewardBuffer_A, r_a_win]; %#ok<AGROW>
                        end
                    end
                    if any(~isnan(rewardB_buffer))
                        r_b_win = mean(rewardB_buffer,'omitnan');
                        if numel(RewardBuffer_B) >= 5
                            RewardBuffer_B = [RewardBuffer_B(2:end), r_b_win];
                        else
                            RewardBuffer_B = [RewardBuffer_B, r_b_win]; %#ok<AGROW>
                        end
                    end

                    T = numel(actions_buffer);
                    advantages_buffer = zeros(1, T, 'like', advantages_buffer_A);

                    idxA = (actions_buffer == 1) & isfinite(advantages_buffer_A);
                    idxB = (actions_buffer == 2) & isfinite(advantages_buffer_B);

                    advantages_buffer(idxA) = advantages_buffer_A(idxA);
                    advantages_buffer(idxB) = advantages_buffer_B(idxB);

                    [dlnet, avgLoss] = updateLSTMActor(dlnet, X_buffer, actions_buffer, ...
                        advantages_buffer, trainingGap, learnRate, temperature, entropyBonus); %#ok<ASGLU>

                    X_buffer = [];
                    actions_buffer = [];
                    rewards_buffer = [];
                    rewardA_buffer = [];
                    rewardB_buffer = [];
                    advantages_buffer_A = [];
                    advantages_buffer_B = [];
                end
            end
        end
    end
end

function Fitness = calFitness(PopObj)
    N = size(PopObj,1);
    if N == 0
        Fitness = [];
        return;
    elseif N == 1
        Fitness = 0;
        return;
    end

    fmax = max(PopObj,[],1);
    fmin = min(PopObj,[],1);
    span = fmax - fmin;
    span(span == 0) = 1;
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(span,N,1);

    Dis = inf(N);
    for i = 1:N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end

function w = feature_weights(~, mask)
    w = ones(sum(mask), 1);
end
function Z = som_grid_points(N)
    if N <= 0
        Z = zeros(0,2);
        return;
    end
    cols = ceil(sqrt(N));
    rows = ceil(N/cols);
    [X,Y] = meshgrid(linspace(0,1,cols),linspace(0,1,rows));
    Z = [X(:),Y(:)];
    Z = Z(1:N,:);
end

% ===================== POP_BANK (all RNG-free => Stage0/1 evolution == v7) =====================

function BANK = updateBank(BANK, Cand, r_frac, K_max, objThr) %#ok<INUSL>
% v13_jointAll: front-1 近前沿档案, 用联合(目标⊕潜)密度维护(取代 v12 潜 niche/migrate)。
%   无分子开关: 准入 = front-1 (有 objThr 时再并上达标, 对 LSMOP 无影响); 累积不同潜向量,
%   超 BANK_CAP 时按联合密度删最冗余(同目标同潜≈同点 先删)。r_frac/K_max 弃用。
    if isempty(Cand); return; end
    keep = (NDSort(Cand.objs, 1) == 1);            % objective-space gate (near-front)
    if ~isempty(objThr)
        keep = keep | (all(Cand.objs <= objThr, 2))';
    end
    Cand = Cand(keep);
    if isempty(Cand); return; end
    if isempty(BANK)
        BANK = Cand(1);
        if numel(Cand) == 1; return; end
        Cand = Cand(2:end);
    end
    Bdec = BANK.decs;                              % [perf] 提取一次, 循环内增量维护
    for i = 1:numel(Cand)
        cdec = Cand(i).dec;
        if any(all(abs(Bdec - cdec) < 1e-12, 2)); continue; end  % 完全相同潜向量(真副本)跳过
        BANK(end+1) = Cand(i);  Bdec(end+1,:) = cdec; %#ok<AGROW>
    end
    BANK_CAP = 100;                                % 限运行时: merge 每代在 union(~pop+CAP) 上选 N
    if numel(BANK) > BANK_CAP
        Del = jointTruncation(BANK.objs, BANK.decs, numel(BANK) - BANK_CAP);
        BANK(Del) = [];
    end
end

function f = dom(a, b) %#ok<DEFNU>                  % does a dominate b ? (保留, 兼容)
    f = all(a <= b) && any(a < b);
end

function Population = injectBank(Population, BANK, inject_max)
% Demand-driven + QUALITY-GATED re-injection of LOST good directions. RNG-free, no FE.
    if isempty(BANK) || inject_max < 1; return; end
    Bobj = BANK.objs;  Pobj = Population.objs;       % quality gate: rep not dominated by pop
    good = false(numel(BANK),1);
    for i = 1:numel(BANK)
        good(i) = ~any(all(Pobj <= Bobj(i,:),2) & any(Pobj < Bobj(i,:),2));
    end
    cand = BANK(good);
    if isempty(cand); return; end
    Pdec = Population.decs;  Bdec = cand.decs;
    lo = min([Pdec;Bdec],[],1);  span = max(max([Pdec;Bdec],[],1)-lo, 1e-12);
    Pn = (Pdec-lo)./span;  Bn = (Bdec-lo)./span;
    Dpp = pdist2(Pn,Pn);  Dpp(1:size(Dpp,1)+1:end) = inf;
    ppNN = min(Dpp,[],2);  refGap = median(ppNN);
    bMin = min(pdist2(Bn,Pn),[],2);
    lost = find(bMin > 2*refGap);                    % truly lost = beyond pop spacing
    if isempty(lost); return; end
    [~,ord] = sort(bMin(lost),'descend');  lost = lost(ord);
    nInj = min([inject_max, numel(lost), numel(Population)]);
    [~,crowd] = sort(ppNN,'ascend');                 % replace most crowded pop points
    Population(crowd(1:nInj)) = cand(lost(1:nInj));
end

function Out = mergeForOutput(Population, BANK, objThr) %#ok<INUSD>
% v13_jointAll: 报告种群 = 在 [pop ∪ BANK] 上做"双空间 SPEA2"选 N (无分子开关)。
%   档案被并回来 -> 找回 pop 丢掉的决策多样近前沿点; 选择用 R(目标支配)+D(联合密度)。
    Out = Population;
    if isempty(BANK); return; end
    Out = jointSelect([Population, BANK], numel(Population));
end

% ===================== 双空间(目标⊕潜)选择 / 密度 (v13_jointAll) =====================
function Out = jointSelect(Pop, N)
% SPEA2 选择, 但密度用联合(目标⊕潜)空间: R 收敛(目标支配强度) + D 联合密度。
    if numel(Pop) <= N; Out = Pop; return; end
    Obj = Pop.objs;  Dec = Pop.decs;
    Fit = calFitnessJoint(Obj, Dec);
    Next = Fit < 1;
    if sum(Next) < N                               % 非支配不足 -> 按 fitness 排名补满
        [~,Rk] = sort(Fit);
        Next = false(size(Fit));
        Next(Rk(1:N)) = true;
    elseif sum(Next) > N                           % 非支配过多 -> 联合密度截断
        idx = find(Next);
        Del = jointTruncation(Obj(idx,:), Dec(idx,:), sum(Next)-N);
        Next(idx(Del)) = false;
    end
    Out = Pop(Next);
end

function Fit = calFitnessJoint(Obj, Dec)
% SPEA2 fitness = R(目标空间被支配强度和) + D(联合密度第 floor(sqrt(N)) 近邻)。
    N = size(Obj,1);
    if N == 1; Fit = 0; return; end
    A = reshape(Obj, N, 1, []);  B = reshape(Obj, 1, N, []);
    Dominate = all(A <= B, 3) & any(A < B, 3);     % Dominate(i,j): i 支配 j (目标空间)
    Dominate(1:N+1:end) = false;
    S = sum(Dominate, 2);                           % i 支配的个数
    R = (S' * Dominate)';                           % 支配 i 的那些 j 的强度和
    Dj = jointDist(Obj, Dec);  Dj(logical(eye(N))) = inf;
    Dj = sort(Dj, 2);
    D  = 1 ./ (Dj(:, floor(sqrt(N))) + 2);
    Fit = (R + D)';                                 % 1×N
end

function Del = jointTruncation(Obj, Dec, K)
% SPEA2式迭代删 K 个联合空间最拥挤者(每删一个重排最近邻向量)。
    n   = size(Obj,1);
    Del = false(n,1);
    if K <= 0 || n == 0; return; end
    D = jointDist(Obj, Dec);  D(logical(eye(n))) = inf;
    while sum(Del) < K
        Remain = find(~Del);
        Temp   = sort(D(Remain,Remain), 2);
        [~,Rk] = sortrows(Temp);
        Del(Remain(Rk(1))) = true;
    end
end

function D = jointDist(Obj, Dec)
% 块平衡联合距离: 目标块/潜块各 min-max 归一化后按维度数取均方(防高维潜淹没低维目标), 开方相加。
% 同目标但潜远 -> D 大(决策多样, 留); 同目标同潜 -> D≈0(几乎同点, 删)。
    On = norm01(Obj);  Dn = norm01(Dec);
    Do = pdist2(On, On, 'squaredeuclidean') / max(size(On,2), 1);
    Dl = pdist2(Dn, Dn, 'squaredeuclidean') / max(size(Dn,2), 1);
    D  = sqrt(Do + Dl);
end

function X = norm01(X)
    lo = min(X,[],1);  span = max(max(X,[],1) - lo, 1e-12);
    X  = (X - lo) ./ span;
end