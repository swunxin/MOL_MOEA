function [ndr, C_BA, C_AB] = ComparePop(Population_A, Population_B)
    % A=旧一代，B=新一代（最小化问题）
    % 返回：
    %   C_BA = 覆盖率 C(B,A)：A 中被 B 覆盖的比例
    %   C_AB = 覆盖率 C(A,B)：B 中被 A 覆盖的比例
    %   ndr  = C_BA - C_AB

    % 合并并(可选)归一化（单调变换不会改变弱支配关系）
    CombinedObjs = [Population_A.objs; Population_B.objs];
    min_vals = min(CombinedObjs, [], 1);
    max_vals = max(CombinedObjs, [], 1);
    range    = max(max_vals - min_vals, eps);     % 防除零
    CombinedObjs = (CombinedObjs - min_vals) ./ range;

    % 规模与索引
    num_A = size(Population_A.objs, 1);
    num_B = size(Population_B.objs, 1);
    indices_A = 1:num_A;
    indices_B = num_A+1 : num_A+num_B;

    % ===== 覆盖率 C(B,A)：A 中被 B 至少一个解弱支配的比例（分母 |A|）=====
    covered_A = 0;
    for j = indices_A                        % 外层：被覆盖集合 A
        for i = indices_B                    % 内层：潜在支配者 B
            if dominates(CombinedObjs(i,:), CombinedObjs(j,:))
                covered_A = covered_A + 1;   % 这个 A 被覆盖，只计一次
                break;
            end
        end
    end
    C_BA = covered_A / num_A;

    % ===== 覆盖率 C(A,B)：B 中被 A 至少一个解弱支配的比例（分母 |B|）=====
    covered_B = 0;
    for j = indices_B                        % 外层：被覆盖集合 B
        for i = indices_A                    % 内层：潜在支配者 A
            if dominates(CombinedObjs(i,:), CombinedObjs(j,:))
                covered_B = covered_B + 1;   % 这个 B 被覆盖，只计一次
                break;
            end
        end
    end
    C_AB = covered_B / num_B;

    % ===== NDR：净覆盖优势 =====
    ndr = C_BA - C_AB;                       % ∈ [-1, 1]
end

function tf = dominates(x, y)
    % 弱支配（最小化）：所有目标不差且至少一维更优
    tf = all(x <= y) && any(x < y);
end