%% ReLSO + SPR: 三维区域点阵分布图
% 设定：
% 1. 地形: 代表潜空间流形
% 2. 三个固定圈: LogP(蓝), SA(绿), QED(紫) -> 代表 ReLSO 基础能力
% 3. 两个移动圈: 亲和力(红), 毒性(黑) -> 代表 SPR 约束效果

clear; clc; close all;

% --- 1. 制造地形数据 ---
[x, y] = meshgrid(linspace(0, 1, 80), linspace(0, 1, 80));
% 地形高度 Z (假设与 QED 相关)
Z_terrain = 0.6 * y.^1.5 + 0.2 * x; 

% --- 2. 辅助函数: 画圈和撒点 ---
% 画圆圈轮廓
plot_circle = @(ax, cx, cy, r, color, style) ...
    plot3(ax, cx + r*cos(linspace(0,2*pi,100)), ...
          cy + r*sin(linspace(0,2*pi,100)), ...
          interp2(x,y,Z_terrain, cx + r*cos(linspace(0,2*pi,100)), cy + r*sin(linspace(0,2*pi,100))) + 0.02, ...
          'Color', color, 'LineWidth', 2, 'LineStyle', style);

% 在圆圈内撒点
scatter_points = @(ax, cx, cy, r, num, color, marker) ...
    scatter3(ax, cx + (r*0.8).*rand(num,1).*cos(2*pi*rand(num,1)), ...
             cy + (r*0.8).*rand(num,1).*sin(2*pi*rand(num,1)), ...
             interp2(x,y,Z_terrain, cx + (r*0.8).*rand(num,1).*cos(2*pi*rand(num,1)), ...
                                   cy + (r*0.8).*rand(num,1).*sin(2*pi*rand(num,1))) + 0.05, ...
             30, marker, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', color);

% --- 3. 开始绘图 ---
f = figure('Color', 'w', 'Position', [100, 100, 1400, 600]);

% ==========================================================
% 左图: ReLSO 基础 + 纠缠 (无 SPR)
% ==========================================================
ax1 = subplot(1, 2, 1);
hold(ax1, 'on'); grid(ax1, 'on'); axis(ax1, 'square'); view(ax1, -20, 50);
title(ax1, '图1: Before SPR (亲和力与毒性重叠)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax1, 'Latent Dim 1'); ylabel(ax1, 'Latent Dim 2'); zlabel(ax1, 'Value');

% 3.1 画地形背景
surf(ax1, x, y, Z_terrain, 'EdgeColor', 'none', 'FaceAlpha', 0.3);
colormap(ax1, gray); shading interp;

% 3.2 画 ReLSO 的三个固定区域 (SA, LogP, QED)
% SA (绿圈 - 易合成)
plot_circle(ax1, 0.2, 0.2, 0.15, 'g', '-');
scatter_points(ax1, 0.2, 0.2, 0.15, 20, 'g', 'o');
text(ax1, 0.2, 0.2, 0.4, 'SA区域', 'Color', 'g', 'FontWeight', 'bold');

% LogP (蓝圈 - 高脂溶)
plot_circle(ax1, 0.8, 0.2, 0.15, 'b', '-');
scatter_points(ax1, 0.8, 0.2, 0.15, 20, 'b', 's');
text(ax1, 0.8, 0.2, 0.4, 'LogP区域', 'Color', 'b', 'FontWeight', 'bold');

% QED (紫圈 - 高类药)
plot_circle(ax1, 0.2, 0.8, 0.15, 'm', '-');
scatter_points(ax1, 0.2, 0.8, 0.15, 20, 'm', '^');
text(ax1, 0.2, 0.8, 0.9, 'QED区域', 'Color', 'm', 'FontWeight', 'bold');

% 3.3 画纠缠的亲和力与毒性 (重叠在右上角)
% 亲和力 (红圈)
plot_circle(ax1, 0.7, 0.7, 0.18, 'r', '--');
% 毒性 (黑圈) - 位置几乎一样
plot_circle(ax1, 0.72, 0.72, 0.18, 'k', '--');

% 撒点: 混杂点 (既是红又是黑)
scatter_points(ax1, 0.71, 0.71, 0.18, 40, 'k', 'd'); 

text(ax1, 0.7, 0.7, 1.0, {'纠缠区', '(高毒+高活)'}, 'Color', 'r', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');


% ==========================================================
% 右图: ReLSO 基础 + 解耦 (有 SPR)
% ==========================================================
ax2 = subplot(1, 2, 2);
hold(ax2, 'on'); grid(ax2, 'on'); axis(ax2, 'square'); view(ax2, -20, 50);
title(ax2, '图2: With SPR (亲和力与毒性分离)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax2, 'Latent Dim 1'); ylabel(ax2, 'Latent Dim 2'); zlabel(ax2, 'Value');

% 4.1 画地形背景 (完全一样)
surf(ax2, x, y, Z_terrain, 'EdgeColor', 'none', 'FaceAlpha', 0.3);
colormap(ax2, gray); shading interp;

% 4.2 画 ReLSO 的三个固定区域 (完全一样，证明基础没变)
plot_circle(ax2, 0.2, 0.2, 0.15, 'g', '-');
scatter_points(ax2, 0.2, 0.2, 0.15, 20, 'g', 'o');
text(ax2, 0.2, 0.2, 0.4, 'SA区域', 'Color', 'g', 'FontWeight', 'bold');

plot_circle(ax2, 0.8, 0.2, 0.15, 'b', '-');
scatter_points(ax2, 0.8, 0.2, 0.15, 20, 'b', 's');
text(ax2, 0.8, 0.2, 0.4, 'LogP区域', 'Color', 'b', 'FontWeight', 'bold');

plot_circle(ax2, 0.2, 0.8, 0.15, 'm', '-');
scatter_points(ax2, 0.2, 0.8, 0.15, 20, 'm', '^');
text(ax2, 0.2, 0.8, 0.9, 'QED区域', 'Color', 'm', 'FontWeight', 'bold');

% 4.3 画 SPR 作用后的新区域 (分开!)

% 亲和力 (红圈) - 移动到了右侧 LogP 附近 (假设亲和力与脂溶性有关)
plot_circle(ax2, 0.85, 0.45, 0.15, 'r', '-');
scatter_points(ax2, 0.85, 0.45, 0.15, 30, 'r', 'p'); % 五角星代表好药
text(ax2, 0.9, 0.45, 0.6, {'理想区域', '(高亲和)'}, 'Color', 'r', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% 毒性 (黑圈) - 被推到了上方 QED 附近
plot_circle(ax2, 0.45, 0.85, 0.15, 'k', '-');
scatter_points(ax2, 0.45, 0.85, 0.15, 20, 'k', 'x'); % 叉号代表毒性
text(ax2, 0.45, 0.9, 1.0, {'危险区域', '(高毒性)'}, 'Color', 'k', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% 画一个 SPR 斥力箭头
quiver3(ax2, 0.65, 0.65, 0.2, -0.2, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver3(ax2, 0.65, 0.65, -0.2, 0.2, 0, 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5);
text(ax2, 0.65, 0.65, 0.8, 'SPR斥力', 'FontSize', 12, 'FontWeight', 'bold');