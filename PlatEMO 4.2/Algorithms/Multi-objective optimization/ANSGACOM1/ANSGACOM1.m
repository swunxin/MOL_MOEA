classdef ANSGACOM1 < ALGORITHM
% <multi/many> <real/integer/label/binary/permutation> <constrained/none>
% ANSGAIII with RL-based selection (SOM vs. Tournament Selection)

%------------------------------- Reference --------------------------------
% H. Jain and K. Deb, An evolutionary many-objective optimization algorithm
% using reference-point based non-dominated sorting approach, part II:
% Handling constraints and extending to an adaptive approach, IEEE
% Transactions on Evolutionary Computation, 2014, 18(4): 602-622.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Generate the reference points and random population
            [Z,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Z = sortrows(Z);
            interval = Z(1,end) - Z(2,end);
            Population = Problem.Initialization();
            Zmin = min(Population(all(Population.cons<=0,2)).objs,[],1);

            %% ====== SOM configuration (CKD-style) ======
            RealN = numel(Population);
            d = max(1, floor(RealN^(1/(Problem.M-1))));
            D_default = repmat(d, 1, Problem.M-1);
            [D,tau0,H,trainingGap,learnRate,temperature,entropyBonus] = Algorithm.ParameterSet( ...
                D_default, 0.7, 5, 10, 0.01, 0.7, 0.05);
            D = max(1, floor(D));
            while prod(D) > RealN
                D = max(1, floor(D * 0.9));
            end
            GridN  = prod(D);
            sigma0 = sqrt(sum(D.^2)/(Problem.M-1))/2;
            [Zgrid, LDis, B] = buildSomGrid(D, H);
            somCfg = struct('D',D,'tau0',tau0,'H',H,'sigma0',sigma0, ...
                            'GridN',GridN,'Zgrid',Zgrid,'LDis',LDis,'B',B);

            %% ====== RL Module Initialization ======
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

            s0_raw   = State(pop0, prev_pi, cfg);
            w0       = feature_weights(0, cfg.feature_mask);
            s0       = s0_raw .* w0;
            inputSize = numel(s0);

            probClip      = [0.1, 0.9];

            outputSize = 2;
            hidden1    = 16;

            layers = [
                sequenceInputLayer(inputSize, "Name","input", "Normalization","none")
                lstmLayer(hidden1, "OutputMode","sequence", "Name","lstm")
                fullyConnectedLayer(outputSize, "Name","fc_out")
            ];
            lgraph = layerGraph(layers);
            dlnet  = dlnetwork(lgraph);

            V_old_A = 0;  V_old_B = 0;
            RewardBuffer_A = []; RewardBuffer_B = [];
            rewardA_buffer = []; rewardB_buffer = [];
            X_buffer = []; actions_buffer = []; rewards_buffer = [];
            advantages_buffer_A = []; advantages_buffer_B = [];
            trainingGapCounter = 0;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                Fitness = calFitness(Population.objs);
                cfg.current_fitness = Fitness;

                progress  = min(max(Problem.FE/Problem.maxFE,0),1);
                state_raw = State(Population, prev_pi, cfg);
                w_t       = feature_weights(progress, cfg.feature_mask);
                state_t   = state_raw .* w_t;
                X_buffer  = [X_buffer, state_t];

                [action, pi_vec] = predictLSTMAction(dlnet, X_buffer, temperature, probClip);
                Population_temp = Population;

                if action == 1
                    MatingPool = TournamentSelection(2,Problem.N,sum(max(0,Population.cons),2));
                else
                    MatingPool = SOMSelection(Population, Fitness, somCfg, Problem);
                end

                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
                Population = EnvironmentalSelection([Population,Offspring],Problem.N,Z,Zmin);
                Population = Pop_Pool(Population, Population_temp, Offspring, Problem);
                Z          = Adaptive(Population.objs,Z,Problem.N,interval);

                [reward, C_BA, C_AB] = ComparePop(Population_temp, Population);
                State('push_cache', struct('have',true,'ndr_prev2cur',reward,'c_ba',C_BA,'c_ab',C_AB));
                prev_pi = pi_vec(action);

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

                if mod(trainingGapCounter, trainingGap) == 0
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

                    T = numel(actions_buffer);
                    advantages_buffer = zeros(1, T, 'like', advantages_buffer_A);
                    idxA = (actions_buffer == 1) & isfinite(advantages_buffer_A);
                    idxB = (actions_buffer == 2) & isfinite(advantages_buffer_B);
                    advantages_buffer(idxA) = advantages_buffer_A(idxA);
                    advantages_buffer(idxB) = advantages_buffer_B(idxB);

                    [dlnet, ~] = updateLSTMActor(dlnet, X_buffer, actions_buffer, ...
                        advantages_buffer, trainingGap, learnRate, temperature, entropyBonus);

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

%% ========================= Helper Functions =========================
function Fitness = calFitness(PopObj)
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
end

function [Zgrid, LDis, B] = buildSomGrid(D, H)
    D = D(:).';
    Dcells = arrayfun(@(S)1:S, D, 'UniformOutput', false);
    coords = cell(1, numel(D));
    [coords{:}] = ndgrid(Dcells{:});
    Zgrid = zeros(numel(coords{1}), numel(D));
    for i = 1:numel(D)
        Zgrid(:,i) = coords{i}(:);
    end
    LDis = pdist2(Zgrid,Zgrid);
    [~,B] = sort(LDis,2);
    if isempty(B)
        B = [];
    else
        B = B(:,2:min(H+1,end));
    end
end
