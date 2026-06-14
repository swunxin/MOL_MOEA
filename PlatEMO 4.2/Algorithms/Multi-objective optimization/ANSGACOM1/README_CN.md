# ANSGACOM1 算法说明

## 算法概述

ANSGACOM1 是基于 ANSGAIII 的多目标优化算法，融合了 CKD 的强化学习模块与 SOM 选择策略。
算法在每一代中由 RL 动态选择父代选择策略：  
- 动作 1：TournamentSelection（ANSGAIII 原始约束锦标赛）  
- 动作 2：SOM-based Selection（基于决策空间的 SOM 邻域选择）  

核心目标：在目标空间自适应参考向量（ANSGAIII）的同时，引入决策空间的自组织映射信息，实现更强的探索-收敛平衡。

---

## 代码结构

文件目录：`PlatEMO 4.2/Algorithms/Multi-objective optimization/ANSGACOM1/`

- `ANSGACOM1.m`：主算法（ANSGAIII + RL + SOM）  
- `Adaptive.m`：参考向量自适应调整（来自 ANSGAIII）  
- `EnvironmentalSelection.m`：NSGA-III 环境选择（来自 ANSGAIII）  
- `State.m`：RL 状态构造器（来自 CKD）  
- `ComparePop.m`：奖励计算（来自 CKD）  
- `predictLSTMAction.m`：动作预测（来自 CKD）  
- `updateLSTMActor.m`：Actor 更新（来自 CKD）  
- `a_update_critic.m`：Critic 更新（来自 CKD）  

---

## 核心流程

1. **初始化**  
   - 生成参考向量 `Z`（ANSGAIII）  
   - 构建 SOM 网格（决策空间）  
   - 初始化 RL 网络与缓存  

2. **每代迭代**  
   - 计算当前种群 Fitness（SDE-style）  
   - 构造 RL 状态，预测动作  
   - 根据动作生成 `MatingPool`：  
     - 动作 1：TournamentSelection  
     - 动作 2：SOM Selection（基于 SOM 邻域 + Fitness 比较）  
   - `OperatorGA` 生成子代  
   - `EnvironmentalSelection` 筛选新种群  
   - `Adaptive` 调整参考向量  
   - `ComparePop` 计算奖励，更新 RL 网络  

---

## RL 状态与奖励

状态向量（7 维，来自 CKD）：  
1. NDR（净支配率）  
2. C_BA  
3. C_AB  
4. 上一步动作概率 `pi_prev`  
5. PF1_ratio  
6. s4（历史最优收敛差距）  
7. s5（全局多样性）  

奖励：  
`reward = C_BA - C_AB`  
衡量新种群对旧种群的支配优势。

---

## 说明与注意事项

- ANSGACOM1 不依赖 ANSGACOM，实现完全基于 ANSGAIII + CKD 组件重组。  
- SOM 选择使用决策空间信息，负责探索；TournamentSelection 保留 ANSGAIII 原始约束处理。  
- 如果问题无约束，则 TournamentSelection 退化为随机选择。  
- 可通过修改 `ANSGACOM1.m` 中的 `ParameterSet` 调整 SOM 和 RL 超参数。  

---

## 调用示例

在 PlatEMO 中选择算法 `ANSGACOM1` 并运行即可：
```matlab
platemo('algorithm',@ANSGACOM1,'problem',@DTLZ2,'M',3,'D',12)
```
