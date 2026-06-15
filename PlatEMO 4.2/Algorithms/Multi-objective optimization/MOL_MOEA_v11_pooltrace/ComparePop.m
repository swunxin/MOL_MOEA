function [ndr, C_BA, C_AB] = ComparePop(Population_A, Population_B)
    A_objs = Population_A.objs;
    B_objs = Population_B.objs;

    num_A = size(A_objs, 1);
    num_B = size(B_objs, 1);
    if num_A == 0 || num_B == 0
        C_BA = 0;
        C_AB = 0;
        ndr  = 0;
        return;
    end

    CombinedObjs = [A_objs; B_objs];
    min_vals = min(CombinedObjs, [], 1);
    max_vals = max(CombinedObjs, [], 1);
    range = max(max_vals - min_vals, eps);
    CombinedObjs = (CombinedObjs - min_vals) ./ range;

    A = CombinedObjs(1:num_A, :);
    B = CombinedObjs(num_A+1:num_A+num_B, :);

    C_BA = coverage_one(B, A);
    C_AB = coverage_one(A, B);
    ndr  = C_BA - C_AB;
end

function C = coverage_one(Dominators, Targets)
    if isempty(Targets)
        C = 0;
        return;
    end

    covered = false(size(Targets,1), 1);
    for j = 1:size(Targets,1)
        covered(j) = any(all(Dominators <= Targets(j,:), 2) & any(Dominators < Targets(j,:), 2));
    end
    C = mean(covered);
end
