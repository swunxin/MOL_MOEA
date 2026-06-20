function [action, pi_t] = predictLSTMAction(dlnet, state_seq, temperature, probClip)
    if nargin < 3 || isempty(temperature)
        temperature = 1;
    end
    if nargin < 4
        probClip = [];
    end
% predictLSTMAction - 基于最后一步状态输出动作（不裁剪或仅极小 ε）
% state_seq: [inputSize x T]

    % ---- 0) 空序列保护 ----
    if isempty(state_seq)
        pi_t = [0.5, 0.5];
        action = (rand < 0.5) + 1; % 1或2
        return;
    end

    softmax_temp = @(x, temp) exp((x - max(x,[],1))./temp) ./ sum(exp((x - max(x,[],1))./temp), 1);

    % ---- 1) 前向 ----
    dlX = dlarray(reshape(state_seq, size(state_seq,1), 1, []), "CBT");
    dlY = forward(dlnet, dlX);               % [numActions x 1 x T]

    logits_t = dlY(:, 1, end);
    pi_dl    = softmax_temp(logits_t, max(temperature, 1e-6));  % 温度最小保护
    pi_vec   = double(extractdata(pi_dl(:)));

    % ---- 2) 数值保护 & 可选裁剪 ----
    pi_vec(~isfinite(pi_vec)) = 0;
    pi_vec = max(pi_vec, 0);
    if ~isempty(probClip)
        lo = probClip(1); hi = probClip(2);
        pi_vec = max(min(pi_vec, hi), lo);
    end
    s = sum(pi_vec);
    if s <= 0
        pi_vec = ones(numel(pi_vec),1)/numel(pi_vec);
    else
        pi_vec = pi_vec / s;
    end

    % ---- 3) 采样（无统计工具箱也可用）----
    try
        action = randsample(numel(pi_vec), 1, true, pi_vec);  % 1=SDE, 2=SOM
    catch
        % 后备：累积分布采样
        cdf = cumsum(pi_vec);
        r = rand;
        action = find(r <= cdf, 1, 'first');
        if isempty(action), action = numel(pi_vec); end
    end

    pi_t   = pi_vec(:).';
end
