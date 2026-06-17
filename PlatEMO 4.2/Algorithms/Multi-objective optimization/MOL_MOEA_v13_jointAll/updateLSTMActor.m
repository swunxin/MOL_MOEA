function [dlnet, avgLoss] = updateLSTMActor(dlnet, X_buffer, actions_buffer, advantages_buffer, sequenceLength, learnRate, temperature, entropyBonus)
% updateLSTMActor - 用优势(advantages)训练 LSTM 策略网络
%
% Inputs:
%   dlnet              - dlnetwork
%   X_buffer           - [inputSize x T]
%   actions_buffer     - [1 x T], 取值∈{1,2}
%   advantages_buffer  - [1 x T], 未执行动作/无效步应为 0 或 NaN
%   sequenceLength     - 例如 trainingGap
%   learnRate          - 学习率
%   temperature        - softmax 温度
%   entropyBonus       - 熵正则系数
%
% Outputs:
%   dlnet              - 更新后的网络
%   avgLoss            - 平均损失

    avgLoss = 0;
    T_all = numel(actions_buffer);
    numBatches = floor(T_all / sequenceLength);
    if numBatches < 1 || isempty(X_buffer), return; end

    % 将 NaN/Inf 的优势置 0（只对实际执行的动作生效）
    advantages_buffer(~isfinite(advantages_buffer)) = 0;

    for b = 1:numBatches
        idx = (b-1)*sequenceLength + 1 : b*sequenceLength;
        X          = X_buffer(:, idx);           % [C x T]
        actions    = actions_buffer(idx);        % [1 x T]
        advantages = advantages_buffer(idx);     % [1 x T]

        [loss, grad] = dlfeval(@actorLoss, dlnet, X, actions, advantages, temperature, entropyBonus);

        % SGD 更新
        dlnet = dlupdate(@(p,g) p - learnRate*g, dlnet, grad);

        avgLoss = avgLoss + double(gather(extractdata(loss))) / numBatches;
    end
end

% ===== 内部：策略梯度损失（时间平均 + 熵正则，全时刻） =====
function [loss, grad] = actorLoss(dlnet, X, actions, advantages, temperature, entropyBonus)
    temperature = max(temperature, 1e-6);
    % reshape 到 [C B T] 并标注 "CBT"（单序列→B=1）
    X = reshape(X, size(X,1), 1, size(X,2));
    dlX = dlarray(X, "CBT");

    % 前向：输出形状 [numActions × 1 × T]
    dlY = forward(dlnet, dlX);
    T = size(dlY, 3);

    % 时间维度循环
    loss = dlarray(0.0);
    entropySum = dlarray(0.0);

    for t = 1:T
        logits_t = dlY(:,1,t);                   % [numActions×1]

        % 温度 softmax（数值稳定）
        z = logits_t/temperature;
        z = z - max(z);
        pi_t = exp(z) ./ sum(exp(z));           % [numActions×1]

        a_t = actions(t);
        adv = advantages(t);                     % 可为 0

        if adv ~= 0                               % adv=0 的步等价于不贡献梯度
            log_pi_at = log(pi_t(a_t) + eps);
            loss = loss - log_pi_at * adv;       % policy gradient with advantage
        end

        % 熵（鼓励探索）：对所有时间步平均
        entropySum = entropySum - sum(pi_t .* log(pi_t + eps));
    end

    % 按时间步平均
    loss = loss / T;
    entropyMean = entropySum / T;
    loss = loss - entropyBonus * entropyMean;

    % 反向传播
    grad = dlgradient(loss, dlnet.Learnables);
end
