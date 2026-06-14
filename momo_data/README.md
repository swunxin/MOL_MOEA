# momo_data —— 内部化 + 对齐后的 Task1 lead / warm-start 数据

> 项目目前**只跑 Task1**(两个 run 脚本都是 `OBJECTIVE_MODE=momo_task1`,无 Task2/3 运行入口)。
> 所以这里**只放 Task1 实际用到的两个文件**,把原来散在外部 `MOMO-master-main/momo/data/` 的
> lead 和 oripops 拷进项目内,并**修正 lead 与 oripops 的 mol_id 错位**。

## 文件
- `task1_leads.csv` —— lead 列表(纯 SMILES,每行一个,799 行)= `qed_test.csv` **删掉第 1 行**后取 SMILES。
- `task1_oripops.csv` —— warm-start(SMILES,mol_id,sim,qed)= `QMO_qed_mol800_optsmiles.csv` 原样(内容不变)。

## 为什么删第 1 行(错位根因)
MOMO 用 `pd.read_csv('qed_test.csv').values` 读 lead,**默认把第 1 行当表头吃掉**;而 `qed_test.csv`
**无表头**,于是误吃掉第 1 个真分子 → QMO `mol_id m` 对应的 lead 其实是 `qed_test` **第 m+2 行**。
我们的 run 用 MATLAB `readlines` 读 lead(**不丢任何行**)+ `mol_id = 行号-1` + 按 `mol_id == current_lead_id`
配 oripops,没复现 MOMO 的"吃第 1 行",于是整体**错位一行**:每个 lead 被注入了邻近 lead 的热启分子
(对当前 lead sim≈0.12,永不达标)—— 这就是 Task1 SR 只有 34.5% 的真因。

**对齐做法**:lead 文件删掉第 1 行后,代码里现成的 `mol_id == current_lead_id` 就 100% 对齐
(本地验证:622/622 个达标好点的 sim 与 oripops CSV 完全一致),**无需改匹配逻辑、无需 -1**。

## 代码改了哪(均为 Task1)
- `optimizer1.py`:`momo_qed_dataset_candidates` / `momo_qed_oripops_candidates` → 指向本目录;
  oripops 候选设为唯一来源(去掉旧的 mol200/top200 候选,消除歧义)。匹配仍 `mol_id == lead_id`。
- `run_parallel_momo_task1.sh`(实际在用的 run 脚本,驱动 `no_gui_task1_momo.m`):`TASK1_LEAD_FILE` → `momo_data/task1_leads.csv`。
- 注:`run_parallel.sh` 是另一个较旧的入口(驱动 `no_gui_task1.m`,读法不同),**未改**;若要用它需先确认其 lead 读法再单独对齐。

## 备注
- Task2/Task3:**当前没有任何运行入口触发**(MATLAB 驱动 `no_gui_task{2,3}_momo.m` 存在但没脚本调用)。
  故这里不放它们的数据;将来真要跑 Task2/3 时,需各自按其 lead 文件的表头情况单独对齐后再内部化。
