classdef MOL_MOEA_v17_dirAdv < ALGORITHM
% ======================================================================
%  MOL_MOEA_v17_dirAdv (2026-06-20): = MOL_MOEA_v17_credit, 唯一改动 = 喂给 actor 的
%    优势从"每动作自比TD"换成"跨动作相对优势"(方案Y)。STATE/REWARD/算子/EnvSel/BANK/
%    fc_out=0/关熵 全部逐字不动; 奖励的取值位置与 op 归因一字未改(沿用V16那套AOS credit)。
%  动机(由 v7/v16/v17 trace 实测定论): pi 全程钉死 0.5 的发动机问题 = per-action critic
%    把方向性信号"自比掉"(稳态 V->2rt => A=rt-0.5V->0), 优势只衡量"这动作比它自己平均
%    好多少", 从不编码"SDE 比 SOM 好多少" -> 没有跨动作推力 -> fc_out 撬不开。
%    (fc_out=0 起步=铁律好设计;熵=中值平衡器, 已关; 这两个都不是病根, 病根是优势自比。)
%  改动(只动优势这一层):
%    弃用 a_update_critic 的自比TD优势; 改用 A = q_taken - q_other,
%    其中 q_A/q_B = 各算子"已记录的 op 奖励 reward(取值位置/归因不动)"的慢EMA。
%      - 跨动作: SDE 持续优于 SOM => qA>qB => A>0 推向 SDE; 晚期 SOM 占优则翻号 -> 学相位调度。
%      - 自带置信门控: |A|=|qA-qB|, 两算子难分时≈0 -> pi 留 0.5(保住稳定50/50红利, 非v5a乱抖)。
%      - dirBeta(慢EMA衰减) 比 critic 的 alpha=0.2 慢, 避免重蹈"自比掉"; advScale=量级旋钮。
%  验收(LSMOP基本盘): (A)pi离0.5且"早偏SDE/晚偏SOM"相位结构、跨种子可复现
%    (B)全量 LSMOP1-9 IGD rank≥v16/v17不塌(多峰题最好更好) (C)分子Task1 6h(过闸后才验)。
% ----------------------------------------------------------------------
% (原 v16 头注) == MOL_MOEA_v12_envSPEA2, 唯一改动 = **archive-assisted 环境选择**
%   (温和注入)。把 BANK 并进环境选择候选池: EnvironmentalSelection([Population,Offspring,BANK],N)。
%
%   动机: v14 的 injectBank 是"强制替换"(绕过选择, 每代硬塞决策远点 -> 累积拖慢收敛, LSMOP rank 降)。
%   本版改成"候选入选": BANK 里被选择丢掉的好方向, 只有在 SPEA2 认为值得(非支配 R + 稀疏 D)时
%   才重回种群 -> **由构造伤不了收敛**(选的还是最优 N), 多样性该留的留。注入提议, 选择裁决。
%   无任何额外旋钮(不设末段门控/不设多样性阈值, 跨问题/跨代数都稳健)。
%   updateBank(L0)照旧建档; injectBank(L2)仍 injectOn=0 关; L1 输出合并按 v12 原样(LSMOP 门控自关)。
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
            %% v17: STATE += [survF1, decDiv, progress] (s0 与循环里追加方式必须一致, 否则 inputSize 不符)
            prev_survF1 = 0;                              % 上代优质存活率(初始无)
            prev_decDiv = decisionDiversity(pop0.decs);   % 决策空间 spread(初始种群)
            s0        = [s0_raw .* w0; prev_survF1; prev_decDiv; 0];   % 末尾 +3 维, progress(0)
            inputSize = numel(s0);

            %% v17: reward 复合(NDR + 多样性credit)的状态量
            v17_lambda   = 0.5;      % 多样性credit 权重
            v17_emaDecay = 0.95;     % ΔdecDiv 在线标准化的 EMA 衰减
            emaVar_d     = 0;        % ΔdecDiv 的 EMA 二阶矩(方差代理)
            emaInit_d    = false;

            %% NEW CRITIC state (relative-advantage A2C): per-operator value EMAs qA, qB.
            %  q_A/q_B = slow EMA of each operator's reward = an operator value estimate.
            %  Actor advantage = q_taken - q_other (cross-action): |qA-qB| acts as a
            %  confidence gate, its sign says which operator is better. dirBeta is slower
            %  than the old critic's alpha=0.2 so the directional signal is not averaged away.
            qA = 0; qB = 0; qInitA = false; qInitB = false;
            dirBeta  = 0.98;         % 算子质量慢EMA 衰减(越大越慢/越稳)
            advScale = 3;            % 跨动作优势放大(组件4:量级旋钮; 10过冲撞clip->降到3)

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

            %% Actor training buffers (relative-advantage A2C).
            %  NOTE: the old per-action self-baseline critic (a_update_critic) and its
            %  reward windows (V_old_A/B, RewardBuffer_A/B, reward*_buffer) were REMOVED
            %  as dead code in 2026-06-21 cleanup; they were write-only after dirAdv.
            X_buffer            = [];   % LSTM input sequence (states), one column per gen
            actions_buffer      = [];   % chosen operator per gen (1=SDE, 2=SOM)
            advantages_buffer_A = [];   % cross-action advantage logged when SDE taken
            advantages_buffer_B = [];   % cross-action advantage logged when SOM taken

            trainingGapCounter = 0;

            %% POP_BANK init (B0: anchor initial leads before any selection can drop them)
            if bankOn
                BANK = updateBank(BANK, Population, r_frac, K_max, objThr);
            end
            if bankOn && mergeOut && ~isempty(objThr)
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
                %% v17: 与 s0 一致地末尾追加 [上代survF1, 上代decDiv, progress]
                state_t   = [state_raw .* w_t; prev_survF1; prev_decDiv; progress];
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
                Population = EnvironmentalSelection([Population,Offspring,BANK], Problem.N);  % v16: BANK 并入候选池(archive-assisted), 由选择裁决

                %% L2 inject (risky lever, default OFF): refill LOST good directions into evolving pop
                if bankOn && injectOn
                    Population = injectBank(Population, BANK, inject_max);
                end
                %% L1 output merge (default ON): report best-N of [evolving pop UNION BANK]; evolution unchanged
                if bankOn && mergeOut && ~isempty(objThr)
                    OutputPop = mergeForOutput(Population, BANK, objThr);
                else
                    OutputPop = Population;
                end

                [reward_ndr, C_BA, C_AB] = ComparePop(Population_temp, Population);
                State('push_cache', struct('have',true,'ndr_prev2cur',reward_ndr,'c_ba',C_BA,'c_ab',C_AB));  % state 仍喂纯 NDR

                %% v17: 复合 reward = NDR + (lambda*progress)*z(ΔdecDiv)。
                %  本代 Offspring 全来自 op_t -> survF1/decDiv 归因干净; 选择不扰 decs。
                decDiv_now = decisionDiversity(Population.decs);          % 决策空间 spread(选择后)
                survF1_now = qualSurvival(Offspring, Population);          % 优质存活率(进选择后种群且在front-1)
                dDiv       = decDiv_now - prev_decDiv;                     % ΔdecDiv 归当代动作
                if ~emaInit_d
                    emaVar_d = dDiv^2 + 1e-12; emaInit_d = true;
                else
                    emaVar_d = v17_emaDecay*emaVar_d + (1-v17_emaDecay)*dDiv^2;
                end
                dDivNorm = max(min(dDiv/(sqrt(emaVar_d)+1e-9), 3), -3);    % 在线标准化+截断
                reward   = reward_ndr + (v17_lambda*progress)*dDivNorm;    % 早期≈纯NDR(护IGD), 晚期加多样credit
                prev_decDiv = decDiv_now;                                  % 供下一代 state
                prev_survF1 = survF1_now;

                prev_pi = pi_vec(action);

                % v17_dirAdv: 弃用 a_update_critic 的自比TD优势, 改用跨动作相对优势。
                %   reward(取值/op归因不动)只喂进对应算子的慢EMA; 优势=q_taken-q_other。
                %   双方都有基线才推(否则A_rel=0, pi留0.5); advScale 控制撬开力度。
                % Update the taken operator's value EMA, then form the cross-action
                % advantage A_rel = q_taken - q_other; the untaken arm logs NaN.
                if action == 1   % SDE taken
                    if ~qInitA, qA = reward; qInitA = true; else, qA = dirBeta*qA + (1-dirBeta)*reward; end
                    if qInitA && qInitB, A_rel = qA - qB; else, A_rel = 0; end

                    advantages_buffer_A = [advantages_buffer_A, advScale*A_rel]; %#ok<AGROW>
                    advantages_buffer_B = [advantages_buffer_B, NaN]; %#ok<AGROW>
                else             % SOM taken
                    if ~qInitB, qB = reward; qInitB = true; else, qB = dirBeta*qB + (1-dirBeta)*reward; end
                    if qInitA && qInitB, A_rel = qB - qA; else, A_rel = 0; end

                    advantages_buffer_A = [advantages_buffer_A, NaN]; %#ok<AGROW>
                    advantages_buffer_B = [advantages_buffer_B, advScale*A_rel]; %#ok<AGROW>
                end

                actions_buffer = [actions_buffer, action]; %#ok<AGROW>

                trainingGapCounter = trainingGapCounter + 1;

                % Train the actor every trainingGap gens: scatter the per-arm advantages
                % back to a dense [1 x T] vector, then one policy-gradient step.
                if mod(trainingGapCounter, trainingGap) == 0
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
                    advantages_buffer_A = [];
                    advantages_buffer_B = [];
                end
            end
        end
    end
end

% ===================== v17: AOS credit / 决策空间 量(纯计算, 不碰搜索) =====================
function d = decisionDiversity(decs)
% 决策空间 spread = 各维 std 的均值。
    if isempty(decs) || size(decs,1) < 2
        d = 0; return;
    end
    d = mean(std(decs, 0, 1));
end

function s = qualSurvival(Offspring, Population)
% 优质存活率 = 本代 Offspring 中"挺进选择后种群 且 落在 front-1"的占比。
%   选择不扰动 decs -> ismember 'rows' 精确归因; 每代 Offspring 全来自同一 op。
    s = 0;
    if isempty(Offspring) || isempty(Population), return; end
    offDecs = Offspring.decs;
    popDecs = Population.decs;
    nOff    = size(offDecs,1);
    if nOff == 0, return; end
    [tf, loc] = ismember(offDecs, popDecs, 'rows');
    if ~any(tf), return; end
    [fno,~] = NDSort(Population.objs, size(popDecs,1));
    isF1    = (fno == 1);
    survIdx = loc(tf);
    s = sum(isF1(survIdx)) / nOff;
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

% ===================== POP_BANK (all RNG-free => Stage0/1 evolution == v7) =====================

function BANK = updateBank(BANK, Cand, r_frac, K_max, objThr)
% Dual-gate archive: objective-space NEAR-FRONT + decision-space NEW-NICHE (radius r).
% r-admission IS the diversity maintenance (reps pairwise > r). Within a niche a
% Pareto-better candidate MIGRATES the rep. Deterministic (no rand) -> Stage0/1 == v7.
    if isempty(Cand); return; end
    keep = (NDSort(Cand.objs, 1) == 1);            % objective-space gate (near-front)
    if ~isempty(objThr)
        keep = keep | (all(Cand.objs <= objThr, 2))';  % MOLECULAR: also admit EVERY threshold-passing
    end                                                 % lead (even if dominated) so it is preserved
    Cand = Cand(keep);
    for i = 1:numel(Cand)
        c = Cand(i);
        isLead = ~isempty(objThr) && all(c.obj <= objThr);  % threshold-passing molecular lead
        if isempty(BANK); BANK = c; continue; end
        Bdec = BANK.decs;  cdec = c.dec;
        allD = [Bdec; cdec];
        lo   = min(allD,[],1);  span = max(max(allD,[],1)-lo, 1e-12);
        Bn   = (Bdec - lo)./span;  cn = (cdec - lo)./span;
        d    = sqrt(sum((Bn - cn).^2, 2));         % decision distance to each rep
        r    = r_frac * sqrt(size(Bdec,2));        % niche radius in normalised space
        [dmin, j] = min(d);
        if dmin <= r && dom(c.obj, BANK(j).obj)
            BANK(j) = c;                           % same niche & Pareto-better -> migrate (upgrade)
        elseif isLead || dmin > r
            BANK(end+1) = c;                       % NEW direction, OR a LEAD (a lead is NEVER discarded) %#ok<AGROW>
            if numel(BANK) > K_max
                BANK = pruneClosest(BANK, objThr);
            end
        end
        % non-lead within r and not dominating -> discarded (diversity rule; == v7 on LSMOP)
    end
end

function f = dom(a, b)                              % does a dominate b ?
    f = all(a <= b) && any(a < b);
end

function BANK = pruneClosest(BANK, objThr)
% Over K_max: drop the most crowded UNPROTECTED point (decision space). RNG-free.
% PROTECTION: threshold-passing leads (all(obj<=objThr)) are protected UP TO a hard
% cap LEAD_CAP; 超过 LEAD_CAP 时连达标 lead 也按拥挤度淘汰(否则 BANK 无上限膨胀 ->
% updateBank 每代越来越慢, 分子长跑会被拖垮; 这是与分子树 v16 统一的关键差异)。
% LEAD_CAP=100 远超所需故不伤 SR。objThr=[] (LSMOP/default) -> 无保护(原行为, LEAD_CAP不激活)。
    LEAD_CAP = 100;
    n = numel(BANK);
    if n == 0; return; end
    if ~isempty(objThr)
        prot = all(BANK.objs <= objThr, 2);
    else
        prot = false(n,1);
    end
    if n > LEAD_CAP
        cand = (1:n)';                               % 超硬顶: 全体可淘汰(含 lead)
    else
        cand = find(~prot);
    end
    if isempty(cand); return; end                    % 未超顶且全保护 -> 保留累积
    Bdec = BANK.decs;
    lo = min(Bdec,[],1);  span = max(max(Bdec,[],1)-lo, 1e-12);
    Bn = (Bdec - lo)./span;
    Dm = pdist2(Bn, Bn);  Dm(1:n+1:end) = inf;
    nn = min(Dm,[],2);                               % nearest-neighbour distance
    [~,kk] = min(nn(cand));                          % most crowded among droppable
    BANK(cand(kk)) = [];
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

function Out = mergeForOutput(Population, BANK, objThr)
% DOMINANCE-BASED REPLACEMENT (size N kept; RNG-free; archive untouched).
% A BANK point is merged into the output ONLY IF either condition holds:
%   (a) DOMINANCE  : it dominates a current point -> upgrade that point to its
%                    dominator. Strictly non-worsening; if no BANK point dominates
%                    anything (converged case) the output == Population == v7.
%   (b) THRESHOLD  : all(obj <= objThr) -> it meets the molecular "good enough" bar
%                    -> admit it (replace the current worst). DISABLED when objThr=[]
%                    (the LSMOP default) -> only (a) is active -> LSMOP == v7.
% No crowding truncation (that was not IGD-optimal and caused tiny easy-problem drift).
    Out = Population;
    if isempty(BANK); return; end
    thrOn = ~isempty(objThr);
    Pobj  = Out.objs;
    for i = 1:numel(BANK)
        bo  = BANK(i).obj;
        dom = find(all(bo <= Pobj,2) & any(bo < Pobj,2));     % current points b dominates
        if ~isempty(dom)
            [~,k] = max(sum(Pobj(dom,:) - bo, 2));            % upgrade the worst dominated one
            idx = dom(k);
            Out(idx) = BANK(i);  Pobj(idx,:) = bo;
        elseif thrOn && all(bo <= objThr)                     % molecular condition (off by default)
            [~,idx] = max(sum(Pobj,2));                       % replace current worst (by obj sum)
            Out(idx) = BANK(i);  Pobj(idx,:) = bo;
        end
    end
end