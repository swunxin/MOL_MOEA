function savePopulationData(trainingGapCounter, Population_A)
    % 提取保存的数据
    Var_save   = Population_A.decs;
    Obj_save   = Population_A.objs;

    % 读取已有数据（如果文件存在）
    if isfile('PS_date.mat')
        load('PS_date.mat', 'data');
    else
        data = struct();
    end

    % 存储数据
    data.(sprintf('iter_%d', trainingGapCounter)).Var = Var_save;
    data.(sprintf('iter_%d', trainingGapCounter)).Obj = Obj_save;

    % 保存到 MAT 文件
    save('PS_date.mat', 'data');
end