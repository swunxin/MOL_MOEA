# momo_data —— 内部化 + 对齐后的 MOMO lead / warm-start 数据

> 目的：把原来散在外部 `MOMO-master-main/momo/data/` 的 lead 和 oripops 拷进项目内,
> 并**修正 lead 与 oripops 的 mol_id 错位**,让代码里现成的 `oripops.mol_id == current_lead_id` 直接对齐。

## 错位根因(已查实)
MOMO 用 `pd.read_csv(lead).values` 读 lead 文件,**默认把第 1 行当表头吃掉**:
- `qed_test.csv`(Task1)**无表头** → 误吃掉第 1 个真分子 → QMO `mol_id m` ↔ 原文件第 `m+2` 行。
- `qeddrd_test.csv`(Task3)**有表头** → 正确吃掉表头 → oripops mol_id 0-based 对齐。

我们的 run 用 MATLAB `readlines` 读 lead(**不丢任何行**)+ `mol_id=行号-1` + `mol_id==current_lead_id` 配 oripops,
没复现 MOMO 的"吃第 1 行",于是 Task1 整体错位一行(注入的热启全是邻近 lead 的,sim≈0.12,永不达标)。

## 对齐做法(本地已验证 100%)
**lead 文件删掉第 1 行、取 SMILES 列**,之后 `mol_id==lead_id`(代码现状,无需 -1)即 100% 对齐:
- `task1_leads.csv` = `qed_test.csv` 删第 1 行(被误吃的真分子),纯 SMILES,799 行。
- `task3_leads.csv` = `qeddrd_test.csv` 删第 1 行(表头),取 SMILES 列,780 行。
- 校验:好点(QED/sim 达标)按 `mol_id==lead_id` 配 lead → Task1 622/622、Task3 823/823 对齐(sim 与 oripops CSV 完全一致)。

## 文件
- `task1_leads.csv` / `task1_oripops.csv`(= QMO_qed_mol800,内容不变,对齐靠 lead 文件) —— **已对齐,可用**
- `task3_leads.csv` / `task3_oripops.csv`(= QMO_qeddrd_mol200) —— **已对齐,可用**(代码暂未接 Task3)
- `task2_logp_test_RAW.csv` + `task2_oripops_plogp_RAW/` —— **原样未对齐**;Task2 是空格分隔 lead + oripops 按区间拆成 6 个文件,管线特殊,需单独对齐后再用。

## 代码改了哪
- `optimizer1.py`:`momo_qed_dataset_candidates` / `momo_qed_oripops_candidates` → 指向本目录(Task1)。匹配逻辑保持 `mol_id==lead_id`(数据已对齐,不再需要 -1)。
- `run_parallel_momo_task1.sh`:`TASK1_LEAD_FILE` → `momo_data/task1_leads.csv`。
- Task2/Task3 的代码路径暂未改(Task2 待对齐;Task3 代码未接)。
