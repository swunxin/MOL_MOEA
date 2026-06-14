classdef MOMONSGAII < ALGORITHM
% <multi/many> <real> <constrained/none>
% MOMO-style NSGA-II (latent-space friendly)
%
% This implementation mimics the main evolutionary logic used in MOMO's
% NSGA2.py: tournament selection by rank+crowding distance, BLX-alpha style
% linear crossover, and Gaussian disturbance mutation.

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
            %% Parameters (kept close to MOMO defaults)
            % pc    : crossover probability
            % alpha : BLX-alpha expansion (MOMO uses d)
            % pm    : mutation probability
            % m     : number of mutated dimensions (MOMO uses m)
            % sigma : Gaussian noise std for disturbance
            [pc,alpha,pm,m,sigma] = Algorithm.ParameterSet(1,0.25,0.1,2,0.0025);

            %% Initialize population
            Population          = Problem.Initialization();
            [FrontNo,CrowdDis]  = MOMONSGAII.RankAndDistance(Population,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % MOMO uses rank + crowding for selection (one-on-one tournament).
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);

                ParentDec  = Population(MatingPool).decs;
                OffDec     = MOMONSGAII.OperatorMOMO(ParentDec,Problem.lower,Problem.upper,pc,alpha,pm,m,sigma);
                Offspring  = Problem.Evaluation(OffDec);

                [Population,FrontNo,CrowdDis] = MOMONSGAII.EnvironmentalSelectionMOMO([Population,Offspring],Problem.N);
            end
        end
    end

    methods(Static,Access=private)
        function OffDec = OperatorMOMO(ParentDec,lower,upper,pc,alpha,pm,m,sigma)
            % OperatorMOMO - BLX-alpha style linear crossover + Gaussian disturbance.

            OffDec = ParentDec;
            [N,D]  = size(OffDec);

            % Ensure bounds are row vectors
            lower = reshape(lower,1,[]);
            upper = reshape(upper,1,[]);
            if numel(lower) ~= D || numel(upper) ~= D
                error('Problem.lower/upper dimension mismatch: expected %d.',D);
            end

            %% Linear (BLX-alpha-like) crossover
            % child = child + r*(mother - child), where r ~ U(-alpha, 1+alpha)
            for i = 1:N
                if rand < pc
                    mother = ParentDec(randi(N),:);
                    r      = (-alpha) + (1 + 2*alpha)*rand;
                    OffDec(i,:) = OffDec(i,:) + r*(mother - OffDec(i,:));
                end
            end

            % Bound repair (also handled by Problem.CalDec, but keep local too)
            OffDec = min(max(OffDec,repmat(lower,N,1)),repmat(upper,N,1));

            %% Gaussian disturbance mutation (mutate m randomly chosen dimensions)
            m = max(1,round(m));
            for i = 1:N
                if rand < pm
                    k   = min(m,D);
                    pos = randperm(D,k);
                    OffDec(i,pos) = OffDec(i,pos) + sigma.*randn(1,k);
                end
            end

            OffDec = min(max(OffDec,repmat(lower,N,1)),repmat(upper,N,1));
        end

        function [Population,FrontNo,CrowdDis] = EnvironmentalSelectionMOMO(Population,N)
            % Same environmental selection as NSGA-II.
            [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
            Next = FrontNo < MaxFNo;

            CrowdDis = CrowdingDistance(Population.objs,FrontNo);

            Last     = find(FrontNo==MaxFNo);
            [~,Rank] = sort(CrowdDis(Last),'descend');
            Next(Last(Rank(1:N-sum(Next)))) = true;

            Population = Population(Next);
            FrontNo    = FrontNo(Next);
            CrowdDis   = CrowdDis(Next);
        end

        function [FrontNo,CrowdDis] = RankAndDistance(Population,N)
            FrontNo  = NDSort(Population.objs,Population.cons,N);
            CrowdDis = CrowdingDistance(Population.objs,FrontNo);
        end
    end
end
