function saveWeightsRewards(trainingGapCounter, w_A, w_B, Reward)
    % 读取已有数据（如果文件存在）
    if isfile('PS_date.mat')
        load('PS_date.mat', 'data');
    else
        data = struct();
    end

    % 存储权重和奖励数据
    data.(sprintf('iter_%d', trainingGapCounter)).w_A = w_A;
    data.(sprintf('iter_%d', trainingGapCounter)).w_B = w_B;
    data.(sprintf('iter_%d', trainingGapCounter)).Reward = Reward;

    % 保存到 MAT 文件
    save('PS_date.mat', 'data');
end