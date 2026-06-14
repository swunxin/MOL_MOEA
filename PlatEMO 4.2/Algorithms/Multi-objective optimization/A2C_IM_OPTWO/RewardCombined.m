function reward = RewardCombined(PopOld, PopNew, opts)
% 奖励 = 0.8 * 净支配优势 + 0.2 * ΔHV(保符号对数压缩并归一化)
% - 净支配优势：调用你已有的 ComparePop(PopOld, PopNew)，返回 B对A 的净支配优势 ∈ [-1,1]
% - ΔHV：通过 HvPair(old,new,...) 得到 dHV = HV_new - HV_old
%
% 可选参数（opts，不传用默认）：
%   opts.w_dom     = 0.8;      % 净支配优势权重
%   opts.w_hv      = 0.2;      % HV项权重
%   opts.hvMargin  = 0.10;     % 归一化参考点外扩比例
%   opts.hvMethod  = 'exact';  % 'exact' 或 'mc'
%   opts.samples   = 2000;     % hvMethod='mc' 时的采样数
%   opts.logScale  = 1e5;      % ΔHV 对数压缩尺度（建议 1e4~1e6）
%
% 返回：
%   reward  - 最终奖励标量
%   comp    - 结构体，便于调试 {dom_adv, dHV, hv_term, w_dom, w_hv}

    if nargin < 3 || isempty(opts), opts = struct; end
    if ~isfield(opts,'w_dom'),     opts.w_dom    = 0.8;   end
    if ~isfield(opts,'w_hv'),      opts.w_hv     = 0.2;   end
    if ~isfield(opts,'hvMargin'),  opts.hvMargin = 0.10;  end
    if ~isfield(opts,'hvMethod'),  opts.hvMethod = 'exact'; end
    if ~isfield(opts,'samples'),   opts.samples  = 2000;  end
    if ~isfield(opts,'logScale'),  opts.logScale = 1e5;   end

    % --- 1) 净支配优势：直接用你现有的 ComparePop(A,B) ---
    dom_adv = ComparePop(PopOld, PopNew);  % ∈ [-1,1]

    % --- 2) ΔHV：与 ComparePop(HV口径)一致（统一归一化→PF1→参考点全1） ---
    hvopts = struct('hvMargin', opts.hvMargin, 'method', opts.hvMethod, 'samples', opts.samples);
    [~, ~, dHV, ~] = HvPair(PopOld.objs, PopNew.objs, hvopts);

    % --- 3) 保符号对数压缩 + 归一化到约 [-1,1] ---
    if dHV > 0
        hv_term =  log1p(opts.logScale * dHV);
    elseif dHV < 0
        hv_term = -log1p(opts.logScale * abs(dHV));
    else
        hv_term = 0;
    end
    hv_term = hv_term / log1p(opts.logScale);  % 归一化

    % --- 4) 线性组合 ---
    reward = opts.w_dom * dom_adv + opts.w_hv * hv_term;

   
end
