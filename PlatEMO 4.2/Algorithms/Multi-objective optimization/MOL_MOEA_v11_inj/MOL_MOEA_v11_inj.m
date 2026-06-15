classdef MOL_MOEA_v11_inj < ALGORITHM
% MOL_MOEA_v11_inj: == MOL_MOEA_v10_bank（含全部达标保护：阈值候选门 / niche永不丢达标 / pruneClosest保护），
%   但 L2 注入默认开(injectOn=1)：每代把 BANK 里"离当前(已塌缩)种群>2×间距"的多样好点注回演化、替换最拥挤点
%   -> 抗塌缩，目标是把"唯一达标分子数"从 v10_bank 的 ~8 提上去（对标 FRCSO ~13.5）。
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
                repmat(ceil(Problem.N.^(1/(Problem.M-1))),1,Problem.M-1), 0.7, 5, 0, 0.01, 1, 1, 1, 5, 0.1, 30, 0);   % injectOn=1 (L2 抗塌缩注入开); 其余同 v10_bank
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

                Population = EnvironmentalSelection([Population,Offspring],V,(Problem.FE/Problem.maxFE)^2);
                Population = Pop_Pool(Population, Population_temp, Offspring, Problem);

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
% PROTECTION (Option A): threshold-passing leads (all(obj<=objThr)) are never
% dropped; if every member is protected, BANK is left as-is so leads accumulate
% beyond K_max. objThr=[] (LSMOP/default) -> no protection (legacy behaviour).
    n = numel(BANK);
    if n == 0; return; end
    if ~isempty(objThr)
        prot = all(BANK.objs <= objThr, 2);
    else
        prot = false(n,1);
    end
    cand = find(~prot);
    if isempty(cand); return; end                    % all protected -> keep all (accumulate)
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