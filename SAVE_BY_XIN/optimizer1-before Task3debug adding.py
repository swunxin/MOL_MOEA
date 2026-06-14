import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import QED
from rdkit.Chem import Descriptors  # for logP
from torch import nn
from tqdm import tqdm
import selfies as sf

from MTLBERT.model import PredictionModel
import build_vocab
from model import ReLSO
import torch.nn.functional as F
import sys

sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))

from SA_Score.sascorer import calculateScore as calculateSAScore

# TDC Oracle for DRD2, GSK3B, SA prediction (Task3, Task4)
try:
    from tdc import Oracle
    drd2_oracle = Oracle('drd2')
    gsk3b_oracle = Oracle('GSK3B')
    sa_oracle = Oracle('sa')
except ImportError:
    drd2_oracle = None
    gsk3b_oracle = None
    sa_oracle = None
    print("[WARNING] TDC not installed. Task3/Task4 will not be available.")


def normalize_sa(smiles):
    """Normalize SA score to [0, 1] range. Higher is better (easier to synthesize)."""
    if sa_oracle is None:
        return np.nan
    try:
        sa_score = sa_oracle(smiles)
        # SA score is in [1, 10], normalize to [0, 1]
        # Lower SA = easier to synthesize, so we do (10 - SA) / 9
        normalized = (10.0 - sa_score) / 9.0
        return max(0.0, min(1.0, normalized))
    except:
        return np.nan


def penalized_logP(mol):
    """
    Penalized logP: logP(mol) - SA(mol).
    Used in MOMO Task2 and JT-VAE benchmark.
    
    Args:
        mol: RDKit molecule object
    Returns:
        float: penalized logP value, or NaN if mol is None
    """
    if mol is None:
        return np.nan
    try:
        logp = Descriptors.MolLogP(mol)
        sa = calculateSAScore(mol)
        return logp - sa
    except:
        return np.nan


# ============= Task5 (MPO Pioglitazone) Helper Functions =============
# Pioglitazone reference molecule
PIOGLITAZONE_SMILES = 'O=C1NC(=O)SC1Cc3ccc(OCCc2ncc(cc2)CC)cc3'
_pioglitazone_mol = Chem.MolFromSmiles(PIOGLITAZONE_SMILES)
_pioglitazone_fp = AllChem.GetMorganFingerprintAsBitVect(_pioglitazone_mol, 2, nBits=1024) if _pioglitazone_mol else None
_pioglitazone_mw = Descriptors.MolWt(_pioglitazone_mol) if _pioglitazone_mol else 356.44

def calc_pioglitazone_dissimilarity(mol):
    """Calculate dissimilarity to Pioglitazone (Gaussian centered at 0, sigma=0.1).
    Higher score = more dissimilar (which is desired)."""
    if mol is None or _pioglitazone_fp is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    sim = DataStructs.TanimotoSimilarity(_pioglitazone_fp, fp)
    # Gaussian modifier: exp(-(sim - 0)^2 / (2 * 0.1^2)) = exp(-sim^2 / 0.02)
    score = np.exp(-sim**2 / 0.02)
    return score

def calc_mw_score(mol, target_mw=None, sigma=10):
    """Calculate MW score (Gaussian centered at target MW)."""
    if mol is None:
        return 0.0
    if target_mw is None:
        target_mw = _pioglitazone_mw
    mw = Descriptors.MolWt(mol)
    score = np.exp(-(mw - target_mw)**2 / (2 * sigma**2))
    return score

def calc_rotatable_bonds_score(mol, target_rb=2, sigma=0.5):
    """Calculate rotatable bonds score (Gaussian centered at target)."""
    if mol is None:
        return 0.0
    rb = Descriptors.NumRotatableBonds(mol)
    score = np.exp(-(rb - target_rb)**2 / (2 * sigma**2))
    return score


# ============= Task6 (Docking) Helper Functions =============
# Docking score calculation using QuickVina2-GPU
# Note: This requires QuickVina2-GPU to be installed and configured
DOCKING_ENABLED = os.getenv('DOCKING_ENABLED', '0').strip() == '1'
DOCKING_PROTEIN = os.getenv('DOCKING_PROTEIN', 'proteins/4lde.pdbqt')
DOCKING_CENTER_X = float(os.getenv('DOCKING_CENTER_X', '14.444'))
DOCKING_CENTER_Y = float(os.getenv('DOCKING_CENTER_Y', '5.250'))
DOCKING_CENTER_Z = float(os.getenv('DOCKING_CENTER_Z', '-18.278'))

def calc_docking_score(smi):
    """Calculate docking score using QuickVina2-GPU.
    Returns a large positive value (bad) if docking fails.
    Note: Lower (more negative) scores are better."""
    if not DOCKING_ENABLED:
        return 0.0  # Return neutral value if docking is disabled
    try:
        from QuickVinaTwoGPU.docking import calculateDockingScore
        score = calculateDockingScore(
            smi,
            protein_file=DOCKING_PROTEIN,
            lig_dir='QuickVinaTwoGPU/ligand_files',
            out_dir='QuickVinaTwoGPU/output',
            log_dir='QuickVinaTwoGPU/log',
            conf_dir='QuickVinaTwoGPU/config',
            vina_cwd='QuickVinaTwoGPU'
        )
        return score
    except Exception as e:
        print(f"[DOCKING ERROR] {e}")
        return 1000.0  # Return bad score on error


# CHANGE THESE
pop_size = 100  # 对齐 MOMO 简单任务：nPop=100（如需复现原设置可改回 2000）
latent_model_path = 'runs/relso_selfies/version_1/last.ckpt'  # 原始 ReLSO (teacher) ckpt
latent_vocab_file = 'selfies_vocab.txt'
admet_vocab_file = 'allmolgen198_vocab.txt'
optimizer_boundary_path = 'optimizer_boundary_relso_512.pt'  # ReLSO latent 预先计算的边界
max_seq_len = 200
device = 'cpu'  # may need to change device in device.py too
selfies = True

# 目标模式：
# - 'momo_task1': QED + similarity (2目标) - 成功标准: QED>=0.9 AND sim>=0.4
# - 'momo_task2': pLogP + similarity (2目标) - 成功标准: pLogP improvement AND sim>=0.4
# - 'momo_task3': QED + DRD2 + similarity (3目标) - 成功标准: QED>=0.8, DRD2>=0.3, sim>=0.4
# - 'momo_task4': QED + GSK3β + SA_norm + similarity (4目标) - 成功标准: QED>=0.8, GSK3β>=0.5, SA>=0.8, sim>=0.3
# - 'momo_task5': Pioglitazone MPO (Dissim + MW + RB + sim) (4目标) - Guacamol benchmark
# - 'momo_task6': QED + Docking + similarity (3目标) - 成功标准: QED>=0.8, docking<=-10, sim>=0.3
# - 'admet': 使用原本的 ADMET 多目标（保留以便回退）
# 可通过环境变量设置: OBJECTIVE_MODE=momo_task1 / ... / momo_task6
objective_mode = os.getenv('OBJECTIVE_MODE', 'momo_task1').strip().lower()
# 保持向后兼容: momo_taskB 等价于 momo_task1
if objective_mode == 'momo_taskb':
    objective_mode = 'momo_task1'

# -------------------- Small tuning knobs (env-configurable) --------------------
# These are intended as low-risk tips to improve generation quality/diversity
# for the same lead, without changing the overall pipeline.
#
# Lead-local init noise (higher explores farther; lower stays closer to lead)
LEAD_INIT_SIGMA = float(os.getenv('LEAD_INIT_SIGMA', '0.38'))

# Repair decoding strategy (used only in the first decode inside repair)
#   - 'argmax' (deterministic, current behavior)
#   - 'sample' (stochastic token sampling; can reduce many-to-one collapse)
REPAIR_DECODE_STRATEGY = os.getenv('REPAIR_DECODE_STRATEGY', 'argmax').strip().lower()
REPAIR_DECODE_TEMPERATURE = float(os.getenv('REPAIR_DECODE_TEMPERATURE', '1.0'))
REPAIR_DECODE_TOP_K = int(os.getenv('REPAIR_DECODE_TOP_K', '0'))
REPAIR_DECODE_ATTEMPTS = int(os.getenv('REPAIR_DECODE_ATTEMPTS', '1'))
REPAIR_AVOID_EXACT_LEAD = os.getenv('REPAIR_AVOID_EXACT_LEAD', '0').strip() in {'1', 'true', 'yes'}

# Repair mode controls whether we do the decode->encode projection step.
#   - 'always' : current behavior (recommended default)
#   - 'none'   : no projection; evaluate objectives on decoded SMILES from raw embeddings
REPAIR_MODE = os.getenv('REPAIR_MODE', 'always').strip().lower()

# Bounds mode controls what decision-variable bounds we send to PlatEMO.
# For unconditional generation, a single global box worked well.
# For lead-local optimization, global bounds can clip the local neighborhood (or place z0 near an edge),
# which reduces effective exploration and can worsen SR.
#   - 'global'     : use optimizer_boundary_relso.pt bounds (default, current behavior)
#   - 'lead_local' : per-run bounds centered at lead embedding z0
BOUNDS_MODE = os.getenv('BOUNDS_MODE', 'global').strip().lower()
# When BOUNDS_MODE=lead_local, use radius = LEAD_BOUNDS_RADIUS_MULT * LEAD_INIT_SIGMA * (global_ub-global_lb)
LEAD_BOUNDS_RADIUS_MULT = float(os.getenv('LEAD_BOUNDS_RADIUS_MULT', '3.0'))
# Optional: intersect lead-local bounds with global bounds
LEAD_BOUNDS_CLIP_TO_GLOBAL = os.getenv('LEAD_BOUNDS_CLIP_TO_GLOBAL', '0').strip() in {'1', 'true', 'yes'}

# Optional duplicate penalty (discourages identical SMILES in the same batch)
# 0.0 keeps exact objective definition.
DUPLICATE_SMILES_PENALTY = float(os.getenv('DUPLICATE_SMILES_PENALTY', '0.0'))

# -------------------- MOMO Alignment Settings --------------------
# These settings help align ReLSO behavior with MOMO's CDDD-based optimization.
#
# Latent space normalization: transform ReLSO's variable-range latent space to [-1, 1]
# This is crucial because MOMO operates in [-1, 1] bounded space.
LATENT_NORMALIZE_TO_UNIT = os.getenv('LATENT_NORMALIZE_TO_UNIT', '1').strip() in {'1', 'true', 'yes'}

# Beam Search decoding: return top-k candidates and pick first valid one
# MOMO uses beam search with num_top=2, which helps when first decode is invalid.
BEAM_SEARCH_ENABLED = os.getenv('BEAM_SEARCH_ENABLED', '1').strip() in {'1', 'true', 'yes'}
BEAM_SEARCH_NUM_TOP = int(os.getenv('BEAM_SEARCH_NUM_TOP', '5'))

# Decoding temperature for softmax (lower = more deterministic, higher = more diverse)
DECODE_TEMPERATURE = float(os.getenv('DECODE_TEMPERATURE', '1.0'))

# Sigma mode for initialization perturbation:
#   - 'z0_std': sigma = LEAD_INIT_SIGMA * z0.std() (RECOMMENDED for ReLSO)
#   - 'span': sigma = LEAD_INIT_SIGMA * (upper_bound - lower_bound) (original, for CDDD-like spaces)
SIGMA_MODE = os.getenv('SIGMA_MODE', 'z0_std').strip().lower()


# MOMO 数据集（用于回退随机初始化/或选 lead 列表），优先用绝对路径，其次用相对路径
# MOMO Task1 数据集 (QED optimization)
momo_qed_dataset_candidates = [
    '/home/xin/LIU/MOMO-master-main/momo/data/qed_test.csv',
    str((Path(__file__).resolve().parent.parent / 'MOMO-master-main' / 'momo' / 'data' / 'qed_test.csv').as_posix()),
]

# MOMO Task2 数据集 (pLogP optimization)
# 注意：文件格式是空格分隔的 "SMILES pLogP"
momo_plogp_dataset_candidates = [
    '/home/xin/LIU/MOMO-master-main/momo/data/logp_test.csv',
    str((Path(__file__).resolve().parent.parent / 'MOMO-master-main' / 'momo' / 'data' / 'logp_test.csv').as_posix()),
]

# MOMO Task3 数据集 (QED + DRD2 optimization)
momo_qeddrd_dataset_candidates = [
    '/home/xin/LIU/MOMO-master-main/momo/data/qeddrd_test.csv',
    str((Path(__file__).resolve().parent.parent / 'MOMO-master-main' / 'momo' / 'data' / 'qeddrd_test.csv').as_posix()),
]

# MOMO Task2 warm-start initial populations (per-lead optsmiles)
momo_plogp_oripops_candidates = [
    '/home/xin/LIU/MOMO-master-main/momo/data/oripops_plogp/QMO_plogp_mol50_optsmiles.csv',
    str((Path(__file__).resolve().parent.parent / 'MOMO-master-main' / 'momo' / 'data' / 'oripops_plogp' / 'QMO_plogp_mol50_optsmiles.csv').as_posix()),
]

# MOMO Task1 warm-start initial populations (per-lead optsmiles)
momo_qed_oripops_candidates = [
    '/home/xin/LIU/MOMO-master-main/momo/data/oripops_qed/QMO_qed_mol800_optsmiles.csv',
    str((Path(__file__).resolve().parent.parent / 'MOMO-master-main' / 'momo' / 'data' / 'oripops_qed' / 'QMO_qed_mol800_optsmiles.csv').as_posix()),
]

# MOMO Task3 warm-start initial populations (per-lead optsmiles)
momo_qeddrd_oripops_candidates = [
    '/home/xin/LIU/MOMO-master-main/momo/data/oripops_qeddrd/QMO_qeddrd_mol200_optsmiles.csv',
    str((Path(__file__).resolve().parent.parent / 'MOMO-master-main' / 'momo' / 'data' / 'oripops_qeddrd' / 'QMO_qeddrd_mol200_optsmiles.csv').as_posix()),
]

admet_model_path = 'MTL-BERT_model.pt'

# There are some more variables below that can be changed.
# END OF CHANGE THESE


def get_admet_encoder(vocab, additional_model_tokens=None):
    """
        Additional tokens can be added for model_str2num. This may be useful for
        adding additional task-specific tokens like in MTLBERT.
    """
    if additional_model_tokens is None:
        additional_model_tokens = []

    model_str2num = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2,
        '<GLOBAL>': 3,
        '<SEP>': 4,  # if fragment cannot be directly encoded, separate fragment's SMILES encoding
    }

    vocab_str2num = {}
    for i, j in enumerate(vocab):
        vocab_str2num[j] = len(vocab_str2num) + i

    for token in additional_model_tokens:
        model_str2num[token] = len(model_str2num) + len(vocab_str2num)

    return model_str2num, vocab_str2num


class ReLSOOptimizerModel:
    def __init__(self, latent_model_path, admet_model_path, max_seq_len, latent_vocab_file, admet_vocab_file, selfies, device):
        # encoder/decoders
        latent_vocab = build_vocab.load_vocab_from_file(latent_vocab_file)
        admet_vocab = build_vocab.load_vocab_from_file(admet_vocab_file)
        self.latent_model_str2num, self.latent_vocab_str2num = build_vocab.get_single_encoder(latent_vocab)
        self.latent_model_num2str, self.latent_vocab_num2str = build_vocab.get_decoders(self.latent_model_str2num,
                                                                            self.latent_vocab_str2num)

        additional_model_tokens = []
        for i in range(100):
            additional_model_tokens.append(f'<p{i}>')

        self.admet_model_str2num, self.admet_vocab_str2num = get_admet_encoder(admet_vocab, additional_model_tokens)
        self.max_seq_len = max_seq_len

        # loading ADMET prediction model
        self.admet_model = PredictionModel(num_layers=8, d_model=256, dff=256 * 4, num_heads=8,
                                           vocab_size=(len(self.admet_model_str2num) + len(self.admet_vocab_str2num)),
                                           dropout_rate=0.1, reg_nums=10, clf_nums=19, maximum_positional_encoding=300)
        self.admet_model.load_state_dict(torch.load(admet_model_path, map_location=torch.device('cpu'))["model_state_dict"])
        self.admet_model.eval()
        self.admet_model = self.admet_model.to('cpu')

        # 使用原始 ReLSO（teacher）作为潜空间编码器/解码器
        # 注意：latent_model_path 应指向原始 ReLSO 的 ckpt（例如 runs/relso_selfies/...）
        self.latent_model = ReLSO.load_from_checkpoint(latent_model_path)

        self.latent_model.eval()
        self.latent_model.to(device)
        self.selfies = selfies

        self.softmax = nn.Softmax(dim=2)

        self.device = device
        
        # 加载边界用于标准化
        self._global_ub = None
        self._global_lb = None
        self._z_span = None
        
    def set_bounds(self, upper_bound, lower_bound):
        """设置全局边界，用于潜空间标准化到[-1, 1]"""
        self._global_ub = upper_bound.detach().cpu().numpy() if isinstance(upper_bound, torch.Tensor) else np.array(upper_bound)
        self._global_lb = lower_bound.detach().cpu().numpy() if isinstance(lower_bound, torch.Tensor) else np.array(lower_bound)
        self._z_span = self._global_ub - self._global_lb
        self._z_span[self._z_span == 0] = 1.0  # 避免除零
        
    def normalize_z(self, z: np.ndarray) -> np.ndarray:
        """将原始潜向量标准化到[-1, 1]区间（类似MOMO的CDDD空间）"""
        if self._global_ub is None:
            return z
        # z_norm = 2 * (z - lb) / (ub - lb) - 1  =>  maps [lb, ub] to [-1, 1]
        z_norm = 2 * (z - self._global_lb) / self._z_span - 1
        return np.clip(z_norm, -1, 1)
    
    def denormalize_z(self, z_norm: np.ndarray) -> np.ndarray:
        """将[-1, 1]标准化空间的向量还原回原始潜空间"""
        if self._global_ub is None:
            return z_norm
        # z = (z_norm + 1) / 2 * (ub - lb) + lb
        z = (z_norm + 1) / 2 * self._z_span + self._global_lb
        return z

    def encode_seq(self, seq, latent_model=True):
        if latent_model:
            model_str2num = self.latent_model_str2num
            vocab_str2num = self.latent_vocab_str2num
        else:
            model_str2num = self.admet_model_str2num
            vocab_str2num = self.admet_vocab_str2num

        if self.selfies:
            encoding = build_vocab.selfies_encode_molecule(seq, model_str2num, vocab_str2num)
        else:
            encoding = build_vocab.smiles_encode_molecule(seq, model_str2num, vocab_str2num)

        encoding = [model_str2num['<GLOBAL>']] + encoding
        encoding = torch.from_numpy(np.array(encoding)).to(device)
        encoding = F.pad(encoding, pad=(0, self.max_seq_len - len(encoding)), value=0)
        return encoding

    def seq_to_emb(self, seq):
        encoding = self.encode_seq(seq)
        encoding = encoding.unsqueeze(0)
        with torch.no_grad():
            z_rep, _ = self.latent_model.encode(encoding)
        return z_rep

    def emb_to_seq(self, z_rep):
        with torch.no_grad():
            self.latent_model = self.latent_model.to(self.device)
            z_rep = z_rep.to(self.device)
            out = self.latent_model.decode(z_rep).permute(0, 2, 1)
            out = self.softmax(out)
            out = torch.argmax(out, dim=2).tolist()
            if self.selfies:
                return [sf.decoder(build_vocab.selfies_decode_molecule(num_arr, self.latent_model_num2str, self.latent_vocab_num2str)) for
                        num_arr in
                        out]
            else:
                return [build_vocab.smiles_decode_molecule(num_arr, self.latent_model_num2str, self.latent_vocab_num2str) for num_arr in
                        out]

    def emb_to_seq_beam_search(self, z_rep: torch.Tensor, num_top: int = 5, temperature: float = 1.0):
        """
        Beam Search解码：返回每个潜向量的top-k候选SMILES列表。
        类似MOMO的CDDD解码策略，优先返回有效分子。
        
        Returns:
            list of list: 每个z对应num_top个候选SMILES
        """
        with torch.no_grad():
            self.latent_model = self.latent_model.to(self.device)
            z_rep = z_rep.to(self.device)
            logits = self.latent_model.decode(z_rep).permute(0, 2, 1)  # [B, L, V]
            
            # 应用温度
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=2)
            
            B, L, V = probs.shape
            results = []
            
            for b in range(B):
                # 对每个样本，获取每个位置的top-k token
                topk_probs, topk_ids = torch.topk(probs[b], k=min(num_top, V), dim=1)  # [L, k]
                
                # 简化beam search：取top-k个最可能的序列（基于累积概率）
                candidates = []
                
                # 方法1：argmax (最可能)
                argmax_seq = torch.argmax(probs[b], dim=1).tolist()
                if self.selfies:
                    smi = sf.decoder(build_vocab.selfies_decode_molecule(argmax_seq, self.latent_model_num2str, self.latent_vocab_num2str))
                else:
                    smi = build_vocab.smiles_decode_molecule(argmax_seq, self.latent_model_num2str, self.latent_vocab_num2str)
                candidates.append(smi)
                
                # 方法2：随机采样生成更多候选（增加多样性）
                for i in range(num_top - 1):
                    sampled = torch.multinomial(probs[b], num_samples=1).squeeze().tolist()
                    if self.selfies:
                        smi = sf.decoder(build_vocab.selfies_decode_molecule(sampled, self.latent_model_num2str, self.latent_vocab_num2str))
                    else:
                        smi = build_vocab.smiles_decode_molecule(sampled, self.latent_model_num2str, self.latent_vocab_num2str)
                    if smi not in candidates:  # 避免重复
                        candidates.append(smi)
                
                results.append(candidates)
            
            return results
    
    def decode_with_validity_fallback(self, z_rep: torch.Tensor, num_top: int = 5, temperature: float = 1.0):
        """
        解码并返回有效分子，若第一个候选无效则尝试其他候选。
        这是MOMO成功的关键策略之一。
        
        Returns:
            list of tuple: [(mol, smiles), ...] 每个z返回一个(mol, smiles)元组
        """
        all_candidates = self.emb_to_seq_beam_search(z_rep, num_top=num_top, temperature=temperature)
        results = []
        
        for candidates in all_candidates:
            found_valid = False
            for smi in candidates:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    results.append((mol, Chem.MolToSmiles(mol)))  # 规范化SMILES
                    found_valid = True
                    break
            if not found_valid:
                # 所有候选都无效，返回第一个
                results.append((None, candidates[0] if candidates else ''))
        
        return results

    def emb_to_seq_with_strategy(
        self,
        z_rep: torch.Tensor,
        strategy: str = 'argmax',
        temperature: float = 1.0,
        top_k: int = 0,
        generator: Optional[torch.Generator] = None,
    ):
        """Decode embeddings with either deterministic argmax or stochastic sampling.

        This is intentionally used only for *repair* (decode->encode projection)
        to reduce many-to-one collapse. Objective evaluation can remain stable.
        """

        strategy = (strategy or 'argmax').strip().lower()
        if strategy not in {'argmax', 'sample'}:
            strategy = 'argmax'

        with torch.no_grad():
            self.latent_model = self.latent_model.to(self.device)
            z_rep = z_rep.to(self.device)
            logits = self.latent_model.decode(z_rep).permute(0, 2, 1)  # [B, L, V]

            if strategy == 'argmax':
                probs = self.softmax(logits)
                token_ids = torch.argmax(probs, dim=2)
            else:
                temp = float(temperature) if temperature and temperature > 0 else 1.0
                scaled = logits / temp
                probs = torch.softmax(scaled, dim=2)

                if isinstance(top_k, int) and top_k > 0 and top_k < probs.shape[2]:
                    topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=2)
                    masked = torch.zeros_like(probs)
                    masked.scatter_(2, topk_idx, topk_vals)
                    probs = masked
                    denom = probs.sum(dim=2, keepdim=True)
                    probs = torch.where(denom > 0, probs / denom, probs)

                B, L, V = probs.shape
                flat = probs.reshape(B * L, V)
                token_ids = torch.multinomial(flat, num_samples=1, replacement=True, generator=generator)
                token_ids = token_ids.reshape(B, L)

            token_ids = token_ids.tolist()
            if self.selfies:
                return [sf.decoder(build_vocab.selfies_decode_molecule(arr, self.latent_model_num2str, self.latent_vocab_num2str)) for arr in token_ids]
            else:
                return [build_vocab.smiles_decode_molecule(arr, self.latent_model_num2str, self.latent_vocab_num2str) for arr in token_ids]

    def properties(self, seqs):
        encodings = torch.stack([self.encode_seq(seq) for seq in seqs]).to(self.device)
        with torch.no_grad():
            return self.admet_model(encodings)


def get_optimizer():
    return ReLSOOptimizerModel(latent_model_path, admet_model_path, max_seq_len, latent_vocab_file,
                               admet_vocab_file, selfies, device)


def read_matrix_file(file_path):
    with open(file_path, 'r') as f:
        numbers = []
        for line in f.readlines():
            numbers.append([])
            for number in line.split(','):
                try:
                    number = float(number)
                except ValueError:
                    print(number)
                    raise Exception
                numbers[-1].append(number)
    return numbers


def write_matrix_file(tmp_file_path, file_path, content):
    with open(tmp_file_path, 'w') as f:
        if type(content[0]) is list:
            for arr in content:
                arr = [str(n) for n in arr]
                f.write(','.join(arr) + '\n')
        elif type(content[0]) is float:
            arr = [str(n) for n in content]
            f.write(','.join(arr) + '\n')
        else:
            raise Exception(f"Not sure how to write {content} as matrix file")
    os.rename(tmp_file_path, file_path)


def read_text_file(file_path: Path) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def morgan_fingerprint(mol):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto_similarity(mol, fp0) -> float:
    if mol is None or fp0 is None:
        return 0.0
    fp = morgan_fingerprint(mol)
    if fp is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp0, fp))


def convert_obj(objs, objs_used, obj_optim_type):
    """
        Convert objectives that are maximization to minimization by multiplying by -1.
        PlatEMO optimization algorithms assume minimization, so we comply by doing so.
    """
    max_objs = np.argwhere(obj_optim_type[objs_used] == 'max').squeeze()
    objs[:, max_objs] *= -1
    return objs


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    # all_clf_heads and all_reg_heads from the trained MTL-BERT in the exact ordering as it was trained
    all_clf_heads = ['pampa_ncats', 'hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins',
                     'cyp2c19_veith', 'cyp2d6_veith', 'cyp3a4_veith', 'cyp1a2_veith', 'cyp2c9_veith',
                     'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels',
                     'cyp3a4_substrate_carbonmangels', 'AMES', 'DILI', 'Skin_Reaction', 'Carcinogens_Lagunin',
                     'ClinTox', 'hERG']
    all_reg_heads = ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb',
                     'hydrationfreeenergy_freesolv', 'ppbr_az', 'vdss_lombardo',
                     'half_life_obach', 'clearance_hepatocyte_az',
                     'clearance_microsome_az', 'LD50_Zhu']
    clf_obj_type = np.array(['max', 'max', 'max', 'max', 'max', 'unk', 'unk', 'unk', 'unk', 'min', 'min', 'min', 'min', 'min', 'min', 'min', 'min', 'min', 'min'])
    reg_obj_type = np.array(['max', 'max', 'max', 'unk', 'min', 'unk', 'unk', 'unk', 'unk', 'min'])
    # Lead 优化场景下默认不使用 docking 目标（昂贵且本需求不需要）
    all_additional_objectives = ['SA_Score']
    optimizer = ReLSOOptimizerModel(latent_model_path, admet_model_path, max_seq_len, latent_vocab_file,
                                    admet_vocab_file, selfies, device)

    ###################################################################
    # VARIABLES BELOW HERE ARE FOR THE MOST PART CHANGEABLE AS NEEDED #
    ###################################################################

    # Paths used for communication with PlatEMO.
    # 自动检测PlatEMO路径（相对于当前脚本位置）
    script_dir = Path(__file__).resolve().parent
    print(f"[DEBUG] script_dir = {script_dir}", flush=True)
    platemo_dir = script_dir / "PlatEMO 4.2"
    print(f"[DEBUG] platemo_dir = {platemo_dir}, exists = {platemo_dir.exists()}", flush=True)
    
    # 如果PlatEMO目录不存在，尝试硬编码路径（兼容旧配置）
    if not platemo_dir.exists():
        platemo_dir = Path("/home/xin/LIU/ManyObjectiveDrugDesign/PlatEMO 4.2")
        print(f"[DEBUG] Using fallback platemo_dir = {platemo_dir}", flush=True)
    
    print(f"Using PlatEMO communication directory: {platemo_dir}", flush=True)
    
    py_EMB_path = platemo_dir / "py_EMB.txt"
    py_EMB_tmp_path = platemo_dir / "py_EMB_tmp.txt"
    py_OBJ_path = platemo_dir / "py_OBJ.txt"
    py_OBJ_tmp_path = platemo_dir / "py_OBJ_tmp.txt"
    py_upper_bound_path = platemo_dir / "py_UPPER.txt"
    py_upper_bound_tmp_path = platemo_dir / "py_UPPER_tmp.txt"
    py_lower_bound_path = platemo_dir / "py_LOWER.txt"
    py_lower_bound_tmp_path = platemo_dir / "py_LOWER_tmp.txt"
    matlab_repair_emb_path = platemo_dir / "matlab_REPAIR_EMB.txt"
    py_M_path = platemo_dir / "py_M.txt"
    py_M_tmp_path = platemo_dir / "py_M_tmp.txt"
    py_N_path = platemo_dir / "py_N.txt"
    py_N_tmp_path = platemo_dir / "py_N_tmp.txt"
    py_init_pop_path = platemo_dir / "py_init_pop.txt"
    py_init_pop_tmp_path = platemo_dir / "py_init_pop_tmp.txt"
    py_shutdown_path = platemo_dir / "py_SHUTDOWN.txt"
    py_new_run_path = platemo_dir / "py_NEW_RUN.txt"

    # Lead communication (plain text, one SMILES)
    py_lead_smiles_path = platemo_dir / "py_LEAD_SMILES.txt"
    # Lead id communication (optional; MOMO datasets use 0-based mol_id)
    py_lead_id_path = platemo_dir / "py_LEAD_ID.txt"

    # Optional: handshake + early-success shortcut (Task1)
    py_run_ready_path = platemo_dir / "py_RUN_READY.txt"
    py_run_ready_tmp_path = platemo_dir / "py_RUN_READY_tmp.txt"
    py_early_success_path = platemo_dir / "py_EARLY_SUCCESS.txt"
    py_early_success_tmp_path = platemo_dir / "py_EARLY_SUCCESS_tmp.txt"

    # MOMO Task1-style record outputs (written next to this script)
    momo_task1_csv_path = script_dir / "MOMO_qed_mol800.csv"
    momo_task1_txt_path = script_dir / "MOMO_qed_mol800.txt"

    if objective_mode in ('momo_task1', 'momo_taskb'):
        # MOMO Task1/TaskB：QED + similarity (2目标)
        reg_objectives = []
        clf_objectives = []
        additional_objectives = []
    elif objective_mode == 'momo_task2':
        # MOMO Task2：pLogP + similarity (2目标)
        # pLogP = logP - SA (penalized logP)
        reg_objectives = []
        clf_objectives = []
        additional_objectives = []
    elif objective_mode == 'momo_task3':
        # MOMO Task3：QED + DRD2 + similarity (3目标)
        reg_objectives = []
        clf_objectives = []
        additional_objectives = []
    elif objective_mode == 'momo_task4':
        # MOMO Task4：QED + GSK3B + SA_norm + similarity (4目标)
        reg_objectives = []
        clf_objectives = []
        additional_objectives = []
    elif objective_mode == 'momo_task5':
        # MOMO Task5：Pioglitazone MPO (4目标)
        reg_objectives = []
        clf_objectives = []
        additional_objectives = []
    elif objective_mode == 'momo_task6':
        # MOMO Task6：QED + Docking + similarity (3目标)
        reg_objectives = []
        clf_objectives = []
        additional_objectives = []
    else:
        reg_objectives = ['LD50_Zhu', 'solubility_aqsoldb']
        clf_objectives = ['bioavailability_ma', 'ClinTox']
        additional_objectives = ['SA_Score']

    print(reg_objectives, clf_objectives)
    # Convert regression objectives to tensor index from ADMET prediction model & task verification
    for i in range(len(reg_objectives)):
        for j in range(len(all_reg_heads)):
            if all_reg_heads[j] == reg_objectives[i]:
                reg_objectives[i] = j
                break
        else:
            raise Exception(f"Unknown regression objective given: {reg_objectives[i]}")

    # Convert classification objectives to tensor index from ADMET prediction model & task verification
    for i in range(len(clf_objectives)):
        for j in range(len(all_clf_heads)):
            if all_clf_heads[j] == clf_objectives[i]:
                clf_objectives[i] = j
                break
        else:
            raise Exception(f"Unknown classification objective given: {clf_objectives[i]}")

    print(reg_objectives, clf_objectives)

    for obj in additional_objectives:
        if obj not in all_additional_objectives:
            raise Exception(f"Unknown additional objective given: {obj}")

    def sa_score(smi):
        try:
            return calculateSAScore(Chem.MolFromSmiles(smi))
        except:
            return 10.0

    upper_bound, lower_bound = torch.load(optimizer_boundary_path, map_location=torch.device('cpu'))  # calculated from generate_optimizer_boundary.py
    
    # 初始化optimizer的边界（用于标准化）
    optimizer.set_bounds(upper_bound, lower_bound)
    
    # 打印配置信息
    print(f"\n{'='*60}")
    print(f"MOMO Alignment Configuration:")
    print(f"  LATENT_NORMALIZE_TO_UNIT: {LATENT_NORMALIZE_TO_UNIT}")
    print(f"  BEAM_SEARCH_ENABLED: {BEAM_SEARCH_ENABLED}")
    print(f"  BEAM_SEARCH_NUM_TOP: {BEAM_SEARCH_NUM_TOP}")
    print(f"  DECODE_TEMPERATURE: {DECODE_TEMPERATURE}")
    print(f"  REPAIR_MODE: {REPAIR_MODE}")
    print(f"  LEAD_INIT_SIGMA: {LEAD_INIT_SIGMA}")
    print(f"  OBJECTIVE_MODE: {objective_mode}")
    print(f"{'='*60}\n")

    # Optional toggle: allow disabling Task1 early-stop for benchmarking consistency
    EARLY_SUCCESS_ENABLED = (os.getenv('EARLY_SUCCESS_ENABLED', '1').strip() == '1')

    # sampling dataset (only used when lead is missing/invalid and we fall back to random init)
    df = None
    if objective_mode in ('momo_task1', 'momo_taskb'):
        for cand in momo_qed_dataset_candidates:
            try_path = Path(cand)
            if try_path.exists():
                df = pd.read_csv(try_path, header=None)[0]
                print(f"Using MOMO QED dataset for fallback init: {try_path}")
                break
        if df is None:
            # 兜底：仍然尝试使用原数据集
            df = pd.read_csv('allmolgen_198max_SMILES_SELFIES_tokenlen.csv')['smiles']
            print("Warning: MOMO QED dataset not found; using allmolgen dataset for fallback init.")
    elif objective_mode == 'momo_task2':
        for cand in momo_plogp_dataset_candidates:
            try_path = Path(cand)
            if try_path.exists():
                df = pd.read_csv(try_path, header=None)[0]
                print(f"Using MOMO pLogP dataset for fallback init: {try_path}")
                break
        if df is None:
            df = pd.read_csv('allmolgen_198max_SMILES_SELFIES_tokenlen.csv')['smiles']
            print("Warning: MOMO pLogP dataset not found; using allmolgen dataset for fallback init.")
    elif objective_mode == 'momo_task3':
        for cand in momo_qeddrd_dataset_candidates:
            try_path = Path(cand)
            if try_path.exists():
                data_df = pd.read_csv(try_path)
                if 'SMILES' in data_df.columns:
                    df = data_df['SMILES']
                else:
                    df = data_df.iloc[:, 0]
                print(f"Using MOMO QED+DRD2 dataset for fallback init: {try_path}")
                break
        if df is None:
            df = pd.read_csv('allmolgen_198max_SMILES_SELFIES_tokenlen.csv')['smiles']
            print("Warning: MOMO QED+DRD2 dataset not found; using allmolgen dataset for fallback init.")
    else:
        df = pd.read_csv('allmolgen_198max_SMILES_SELFIES_tokenlen.csv')['smiles']

    pbar = tqdm()
    if objective_mode in ('momo_task1', 'momo_taskb'):
        print("Objective mode=momo_task1 (QED + similarity)")
        print("  Success criterion: QED >= 0.9 AND sim >= 0.4")
    elif objective_mode == 'momo_task2':
        print("Objective mode=momo_task2 (pLogP + similarity)")
        print("  Success criterion: pLogP improvement AND sim >= 0.4")
        print("  pLogP = logP - SA (penalized logP)")
    else:
        print(f"Total objectives selected: {len(reg_objectives)} reg, {len(clf_objectives)} clf, and {len(additional_objectives)} additional")
    invalid = 0
    total = 0

    # Current run lead cache (set at each py_NEW_RUN)
    current_lead_smiles = None
    current_lead_fp = None
    current_lead_plogp = None  # For task2: store lead's pLogP value
    current_lead_id = None     # Optional: 0-based lead id for MOMO warm-start
    # 局部扰动强度：越大探索越远；建议先用 0.15~0.30
    lead_init_sigma = LEAD_INIT_SIGMA

    # 多样性统计累积（每个run结束后打印）
    diversity_stats = {'unique_total': 0, 'valid_total': 0, 'lead_copies_total': 0, 'eval_count': 0}

    # Cache for MOMO Task2 optsmiles warm-start dataset
    task2_oripops_df = None

    # Cache for MOMO Task1 optsmiles warm-start dataset
    task1_oripops_df = None

    # Cache for MOMO Task3 optsmiles warm-start dataset
    task3_oripops_df = None

    # MOMO Task1 record state (mimics MOMO_task1.py outputs)
    momo_task1_records = []  # (SMILES, mol_id, qed, sim)
    momo_task1_sr = 0

    # Last-evaluated batch cache (used as "final population" proxy)
    last_eval_smiles = None
    last_eval_qed_raw = None
    last_eval_sim_raw = None
    last_eval_mol_id = None
    last_eval_objective_mode = None

    def _write_momo_task1_outputs(mol_id: int):
        try:
            df_out = pd.DataFrame(momo_task1_records, columns=['SMILES', 'mol_id', 'qed', 'sim'])
            df_out.to_csv(momo_task1_csv_path, index=False)
            processed = int(mol_id) + 1  # MOMO: nn-mm1+1 with mm1=0 => nn+1
            np.savetxt(momo_task1_txt_path, [processed, momo_task1_sr], fmt='%s')
        except Exception as e:
            print(f"Warning: failed to write MOMO_task1-style outputs: {e}")

    def _finalize_task1_run_from_last_eval():
        global momo_task1_sr
        global last_eval_smiles, last_eval_qed_raw, last_eval_sim_raw
        global last_eval_mol_id, last_eval_objective_mode

        if last_eval_objective_mode not in ('momo_task1', 'momo_taskb'):
            return
        if last_eval_mol_id is None or last_eval_smiles is None or last_eval_qed_raw is None or last_eval_sim_raw is None:
            return

        # SR: any meets (qed>=0.9 & sim>=0.4)
        try:
            if any((q >= 0.9) and (s >= 0.4) for q, s in zip(last_eval_qed_raw, last_eval_sim_raw)):
                momo_task1_sr += 1
        except Exception:
            pass

        # Record: unique SMILES with qed>=0.9 from the (proxy) final population
        try:
            seen = set()
            for smi, q, s in zip(last_eval_smiles, last_eval_qed_raw, last_eval_sim_raw):
                if not isinstance(smi, str) or len(smi) == 0:
                    continue
                if float(q) < 0.9:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                # MOMO checks uniqueness on the decoded SMILES string directly.
                if smi in seen:
                    continue
                seen.add(smi)
                momo_task1_records.append((smi, int(last_eval_mol_id), float(q), float(s)))
        except Exception as e:
            print(f"Warning: failed to finalize Task1 record from last eval: {e}")

        _write_momo_task1_outputs(int(last_eval_mol_id))

        # Clear cache so we don't double-count
        last_eval_smiles = None
        last_eval_qed_raw = None
        last_eval_sim_raw = None
        last_eval_mol_id = None
        last_eval_objective_mode = None

    def _load_task2_oripops_df():
        global task2_oripops_df
        if task2_oripops_df is not None:
            return task2_oripops_df
        for cand in momo_plogp_oripops_candidates:
            p = Path(cand)
            if p.exists():
                try:
                    task2_oripops_df = pd.read_csv(p)
                    print(f"Using MOMO Task2 oripops warm-start: {p}")
                    return task2_oripops_df
                except Exception as e:
                    print(f"Warning: failed to read Task2 oripops CSV at {p}: {e}")
        task2_oripops_df = pd.DataFrame()
        print("Warning: MOMO Task2 oripops CSV not found; warm-start disabled for this run.")
        return task2_oripops_df

    def _get_task2_oripops_smiles_for_lead(lead_id: int):
        df_orip = _load_task2_oripops_df()
        if df_orip is None or df_orip.empty:
            return []
        if 'mol_id' not in df_orip.columns or 'SMILES' not in df_orip.columns:
            print("Warning: Task2 oripops CSV missing required columns {'SMILES','mol_id'}; warm-start disabled.")
            return []
        try:
            sub = df_orip[df_orip['mol_id'] == lead_id]['SMILES']
            # Keep order as in file; drop missing
            return [s for s in sub.tolist() if isinstance(s, str) and len(s) > 0]
        except Exception as e:
            print(f"Warning: failed to filter Task2 oripops for mol_id={lead_id}: {e}")
            return []

    def _load_task1_oripops_df():
        global task1_oripops_df
        if task1_oripops_df is not None:
            return task1_oripops_df
        for cand in momo_qed_oripops_candidates:
            p = Path(cand)
            if p.exists():
                try:
                    task1_oripops_df = pd.read_csv(p)
                    print(f"Using MOMO Task1 oripops warm-start: {p}")
                    return task1_oripops_df
                except Exception as e:
                    print(f"Warning: failed to read Task1 oripops CSV at {p}: {e}")
        task1_oripops_df = pd.DataFrame()
        print("Warning: MOMO Task1 oripops CSV not found; warm-start disabled for this run.")
        return task1_oripops_df

    def _get_task1_oripops_rows_for_lead(lead_id: int):
        df_orip = _load_task1_oripops_df()
        if df_orip is None or df_orip.empty:
            return None
        required = {'SMILES', 'mol_id', 'sim', 'qed'}
        if not required.issubset(set(df_orip.columns)):
            print(f"Warning: Task1 oripops CSV missing required columns {required}; warm-start disabled.")
            return None
        try:
            sub = df_orip[df_orip['mol_id'] == lead_id]
            if sub is None or sub.empty:
                return None
            return sub
        except Exception as e:
            print(f"Warning: failed to filter Task1 oripops for mol_id={lead_id}: {e}")
            return None

    def _load_task3_oripops_df():
        global task3_oripops_df
        if task3_oripops_df is not None:
            return task3_oripops_df
        for cand in momo_qeddrd_oripops_candidates:
            p = Path(cand)
            if p.exists():
                try:
                    task3_oripops_df = pd.read_csv(p)
                    print(f"Using MOMO Task3 oripops warm-start: {p}")
                    return task3_oripops_df
                except Exception as e:
                    print(f"Warning: failed to read Task3 oripops CSV at {p}: {e}")
        task3_oripops_df = pd.DataFrame()
        print("Warning: MOMO Task3 oripops CSV not found; warm-start disabled for this run.")
        return task3_oripops_df

    def _get_task3_oripops_smiles_for_lead(lead_id: int):
        df_orip = _load_task3_oripops_df()
        if df_orip is None or df_orip.empty:
            return []
        if 'mol_id' not in df_orip.columns or 'SMILES' not in df_orip.columns:
            print("Warning: Task3 oripops CSV missing required columns {'SMILES','mol_id'}; warm-start disabled.")
            return []
        try:
            sub = df_orip[df_orip['mol_id'] == lead_id]['SMILES']
            return [s for s in sub.tolist() if isinstance(s, str) and len(s) > 0]
        except Exception as e:
            print(f"Warning: failed to filter Task3 oripops for mol_id={lead_id}: {e}")
            return []
    
    print("Entering main loop, waiting for MATLAB...", flush=True)
    while True:
        if py_shutdown_path.exists():
            os.remove(py_shutdown_path)
            if total > 0:
                print(f"Invalid solutions {invalid / total * 100}")
            # Finalize last run if needed (Task1 record/SR)
            _finalize_task1_run_from_last_eval()
            print("Found python shutdown file.")
            exit(1)
        elif py_new_run_path.exists():
            # Finalize previous run (MOMO Task1 record/SR) before resetting state.
            _finalize_task1_run_from_last_eval()

            # 打印上一个 run 的多样性统计（如果有）
            if diversity_stats['eval_count'] > 0:
                avg_unique = diversity_stats['unique_total'] / diversity_stats['eval_count']
                avg_valid = diversity_stats['valid_total'] / diversity_stats['eval_count']
                avg_lead = diversity_stats['lead_copies_total'] / diversity_stats['eval_count']
                print(f"\n[RUN SUMMARY] Evaluations: {diversity_stats['eval_count']}, "
                      f"Avg Unique: {avg_unique:.1f}/100, Avg Lead copies: {avg_lead:.1f}", flush=True)
            # 重置统计
            diversity_stats = {'unique_total': 0, 'valid_total': 0, 'lead_copies_total': 0, 'eval_count': 0}
            
            """
            New run file found, so write all necessary information using files so MATLAB can read it.
            """
            if total > 0:
                print(f"Invalid solutions {invalid / total * 100}")
            print(read_matrix_file(py_new_run_path))
            random_state = int(read_matrix_file(py_new_run_path)[0][0])
            print(random_state)

            # 默认每个 run 都清空 lead cache
            current_lead_smiles = None
            current_lead_fp = None
            current_lead_plogp = None  # For task2
            current_lead_id = None     # For task2 warm-start
            # Also clear last-eval cache to prevent cross-run contamination
            last_eval_smiles = None
            last_eval_qed_raw = None
            last_eval_sim_raw = None
            last_eval_mol_id = None
            last_eval_objective_mode = None
            # Clear handshake flags (MATLAB may poll these)
            try:
                if py_run_ready_path.exists():
                    os.remove(py_run_ready_path)
            except Exception:
                pass
            try:
                if py_early_success_path.exists():
                    os.remove(py_early_success_path)
            except Exception:
                pass
            use_similarity_objective = False
            run_lb = None
            run_ub = None

            # 读取 lead id（可选，用于 Task2 的 per-lead warm-start）
            if py_lead_id_path.exists():
                try:
                    lead_id_raw = read_text_file(py_lead_id_path).strip()
                    current_lead_id = int(float(lead_id_raw))
                except Exception as e:
                    print(f"Warning: failed to parse py_LEAD_ID.txt: {e}")
                    current_lead_id = None

            # 读取 lead SMILES（若存在），并计算指纹
            if py_lead_smiles_path.exists():
                lead_raw = read_text_file(py_lead_smiles_path)
                lead_mol = Chem.MolFromSmiles(lead_raw)
                if lead_mol is not None:
                    current_lead_smiles = Chem.MolToSmiles(lead_mol)
                    current_lead_fp = morgan_fingerprint(lead_mol)
                    use_similarity_objective = True
                    # Task2: 计算lead的pLogP作为基准
                    if objective_mode == 'momo_task2':
                        current_lead_plogp = penalized_logP(lead_mol)
                        print(f"Lead pLogP (baseline): {current_lead_plogp:.4f}")

            if current_lead_smiles is not None:
                # MOMO 风格：围绕 z0 高斯扰动产生初代种群
                rng = np.random.RandomState(random_state)
                z0 = optimizer.seq_to_emb(current_lead_smiles).detach().cpu().squeeze().numpy()  # [D]
                z0_std = float(z0.std())  # ReLSO潜空间的实际标准差
                
                global_ub = upper_bound.detach().cpu().numpy()
                global_lb = lower_bound.detach().cpu().numpy()
                span = (global_ub - global_lb)
                span[span == 0] = 1.0

                # Diagnostics: where is z0 relative to the global bounds?
                z0_min_margin = float(np.min(np.minimum((z0 - global_lb) / span, (global_ub - z0) / span)))
                if z0_min_margin < 0:
                    print(f"Warning: lead z0 is outside GLOBAL bounds (min normalized margin={z0_min_margin:.4f}).")
                else:
                    print(f"Lead z0 min normalized margin to GLOBAL bounds: {z0_min_margin:.4f}")
                
                print(f"z0 stats: mean={z0.mean():.6f}, std={z0_std:.6f}")

                # ===== 计算实际sigma =====
                sigma_mode = SIGMA_MODE if SIGMA_MODE in {'z0_std', 'span'} else 'z0_std'
                if sigma_mode == 'z0_std':
                    # 推荐：基于z0的标准差（适合ReLSO这种小范围潜空间）
                    actual_sigma = lead_init_sigma * z0_std
                    print(f"Using SIGMA_MODE=z0_std: actual_sigma = {lead_init_sigma} * {z0_std:.6f} = {actual_sigma:.6f}")
                else:
                    # 原始方式：基于span（适合CDDD这种[-1,1]空间）
                    actual_sigma = lead_init_sigma  # 会在下面乘以span
                    print(f"Using SIGMA_MODE=span: sigma={lead_init_sigma} (will multiply by span)")

                # ===== MOMO Alignment: 标准化潜空间到[-1, 1] =====
                if LATENT_NORMALIZE_TO_UNIT:
                    # 工作在标准化空间中，类似MOMO的CDDD空间
                    z0_norm = optimizer.normalize_z(z0)
                    
                    # 在标准化空间中进行高斯扰动
                    eps = rng.normal(loc=0.0, scale=1.0, size=(pop_size, z0.shape[0])).astype(np.float32)
                    if sigma_mode == 'z0_std':
                        # 标准化空间中也用z0_std相对比例
                        norm_sigma = actual_sigma / z0_std * 0.5  # 归一化后的等效sigma
                        init_pop_norm = z0_norm.reshape(1, -1) + norm_sigma * eps
                    else:
                        init_pop_norm = z0_norm.reshape(1, -1) + lead_init_sigma * eps
                    init_pop_norm = np.clip(init_pop_norm, -1.0, 1.0)  # MOMO风格的硬边界
                    init_pop_norm[0, :] = z0_norm  # 保证lead自身在初代
                    
                    # PlatEMO也工作在[-1, 1]空间
                    run_lb = np.full_like(global_lb, -1.0)
                    run_ub = np.full_like(global_ub, 1.0)
                    init_pop = init_pop_norm.tolist()
                    
                    print(f"Using NORMALIZED latent space [-1, 1] (MOMO-aligned)")
                else:
                    # 原始空间（推荐用于ReLSO + z0_std模式）
                    # Decide per-run bounds to send to PlatEMO
                    if BOUNDS_MODE not in {'global', 'lead_local'}:
                        bounds_mode = 'global'
                    else:
                        bounds_mode = BOUNDS_MODE

                    if bounds_mode == 'lead_local':
                        radius = (LEAD_BOUNDS_RADIUS_MULT * actual_sigma) * np.ones_like(span)
                        run_lb = z0 - radius
                        run_ub = z0 + radius
                        if LEAD_BOUNDS_CLIP_TO_GLOBAL:
                            run_lb = np.maximum(run_lb, global_lb)
                            run_ub = np.minimum(run_ub, global_ub)
                        print(f"Using lead-local bounds mode. radius={radius[0]:.6f}, clip_to_global={LEAD_BOUNDS_CLIP_TO_GLOBAL}")
                    else:
                        run_lb = global_lb
                        run_ub = global_ub

                    eps = rng.normal(loc=0.0, scale=1.0, size=(pop_size, z0.shape[0])).astype(np.float32)
                    if sigma_mode == 'z0_std':
                        # 直接用绝对sigma
                        init_pop_np = z0.reshape(1, -1) + actual_sigma * eps
                    else:
                        # 原始方式：乘以span
                        init_pop_np = z0.reshape(1, -1) + (lead_init_sigma * span.reshape(1, -1)) * eps
                    init_pop_np = np.clip(init_pop_np, run_lb.reshape(1, -1), run_ub.reshape(1, -1))
                    init_pop_np[0, :] = z0  # 保证把 lead 自身放入初代
                    init_pop = init_pop_np.tolist()

                # ===== MOMO Task2 warm-start: mix per-lead optsmiles into init pop =====
                # 默认启用：只要 MATLAB 提供了 py_LEAD_ID.txt 且能找到对应的 optsmi 种群，就会自动注入。
                if objective_mode == 'momo_task2' and current_lead_id is not None:
                    seed_smiles = _get_task2_oripops_smiles_for_lead(current_lead_id)
                    if len(seed_smiles) > 0:
                        max_seeds = pop_size - 1
                        seed_smiles = seed_smiles[:max_seeds]
                        seed_z_list = []
                        for smi in seed_smiles:
                            try:
                                z = optimizer.seq_to_emb(smi).detach().cpu().squeeze().numpy()
                                if LATENT_NORMALIZE_TO_UNIT:
                                    z = optimizer.normalize_z(z)
                                    z = np.clip(z, -1.0, 1.0)
                                else:
                                    z = np.clip(z, run_lb, run_ub)
                                seed_z_list.append(z)
                            except Exception:
                                continue

                        if len(seed_z_list) > 0:
                            if LATENT_NORMALIZE_TO_UNIT:
                                init_arr = np.array(init_pop, dtype=np.float32)
                                for idx, z in enumerate(seed_z_list, start=1):
                                    if idx >= pop_size:
                                        break
                                    init_arr[idx, :] = z
                                init_pop = init_arr.tolist()
                            else:
                                init_arr = np.array(init_pop, dtype=np.float32)
                                for idx, z in enumerate(seed_z_list, start=1):
                                    if idx >= pop_size:
                                        break
                                    init_arr[idx, :] = z
                                init_pop = init_arr.tolist()
                            print(f"Task2 warm-start: injected {len(seed_z_list)} optsmiles seeds for mol_id={current_lead_id}")
                    else:
                        print(f"Task2 warm-start: no optsmiles found for mol_id={current_lead_id}")

                # ===== MOMO Task1 warm-start: mix per-lead optsmiles into init pop =====
                if objective_mode in ('momo_task1', 'momo_taskb') and current_lead_id is not None:
                    sub = _get_task1_oripops_rows_for_lead(current_lead_id)
                    if sub is not None and not sub.empty:
                        seed_smiles = [s for s in sub['SMILES'].tolist() if isinstance(s, str) and len(s) > 0]
                        if len(seed_smiles) > 0:
                            max_seeds = pop_size - 1
                            seed_smiles = seed_smiles[:max_seeds]
                            seed_z_list = []
                            for smi in seed_smiles:
                                try:
                                    z = optimizer.seq_to_emb(smi).detach().cpu().squeeze().numpy()
                                    if LATENT_NORMALIZE_TO_UNIT:
                                        z = optimizer.normalize_z(z)
                                        z = np.clip(z, -1.0, 1.0)
                                    else:
                                        z = np.clip(z, run_lb, run_ub)
                                    seed_z_list.append(z)
                                except Exception:
                                    continue
                            if len(seed_z_list) > 0:
                                init_arr = np.array(init_pop, dtype=np.float32)
                                for idx, z in enumerate(seed_z_list, start=1):
                                    if idx >= pop_size:
                                        break
                                    init_arr[idx, :] = z
                                init_pop = init_arr.tolist()
                                print(f"Task1 warm-start: injected {len(seed_z_list)} optsmiles seeds for mol_id={current_lead_id}")
                    else:
                        print(f"Task1 warm-start: no optsmiles found for mol_id={current_lead_id}")

                # ===== MOMO Task3 warm-start: mix per-lead optsmiles into init pop =====
                # 不复用 Task1/Task2 的选择逻辑，保持 Task3 独立路径，便于稳定对齐与调试。
                if objective_mode == 'momo_task3' and current_lead_id is not None:
                    seed_smiles = _get_task3_oripops_smiles_for_lead(current_lead_id)
                    if len(seed_smiles) > 0:
                        max_seeds = pop_size - 1
                        seed_smiles = seed_smiles[:max_seeds]
                        seed_z_list = []
                        for smi in seed_smiles:
                            try:
                                z = optimizer.seq_to_emb(smi).detach().cpu().squeeze().numpy()
                                if LATENT_NORMALIZE_TO_UNIT:
                                    z = optimizer.normalize_z(z)
                                    z = np.clip(z, -1.0, 1.0)
                                else:
                                    z = np.clip(z, run_lb, run_ub)
                                seed_z_list.append(z)
                            except Exception:
                                continue

                        if len(seed_z_list) > 0:
                            init_arr = np.array(init_pop, dtype=np.float32)
                            for idx, z in enumerate(seed_z_list, start=1):
                                if idx >= pop_size:
                                    break
                                init_arr[idx, :] = z
                            init_pop = init_arr.tolist()
                            print(f"Task3 warm-start: injected {len(seed_z_list)} optsmiles seeds for mol_id={current_lead_id}")
                    else:
                        print(f"Task3 warm-start: no optsmiles found for mol_id={current_lead_id}")

                print(
                    "RUN_CONFIG "
                    f"normalize={LATENT_NORMALIZE_TO_UNIT} "
                    f"sigma_mode={sigma_mode} "
                    f"lead_sigma={float(lead_init_sigma):.4f} "
                    f"actual_sigma={actual_sigma:.6f} "
                    f"repair_mode={REPAIR_MODE} "
                    f"beam_search={BEAM_SEARCH_ENABLED}"
                )
                print(f"Using lead-local init pop. lead={current_lead_smiles}")
            else:
                if objective_mode == 'momo_taskB':
                    raise RuntimeError("objective_mode=momo_taskB requires a valid lead SMILES (py_LEAD_SMILES.txt)")
                # 回退：保持原始“随机采样数据集”初始化
                init_pop = df.sample(pop_size, replace=True, random_state=random_state).tolist()
                init_pop = [optimizer.seq_to_emb(seq).squeeze().tolist() for seq in init_pop]
                print("Lead not provided/invalid; using random dataset init pop.")

            # ===== Write files in order MATLAB expects =====
            # MATLAB DDProblem1.Setting() reads: py_M -> py_N -> py_LOWER -> py_UPPER
            # MATLAB DDProblem1.Initialization() reads: py_init_pop (LAST)
            
            print(f"[DEBUG] Writing py_M.txt and py_N.txt...", flush=True)
            if objective_mode in ('momo_task1', 'momo_taskb', 'momo_task2'):
                M_total = 2
            elif objective_mode in ('momo_task3', 'momo_task6'):
                M_total = 3
            elif objective_mode in ('momo_task4', 'momo_task5'):
                M_total = 4
            else:
                M_total = len(reg_objectives) + len(clf_objectives) + len(additional_objectives)
                if use_similarity_objective:
                    M_total += 1
            write_matrix_file(py_M_tmp_path, py_M_path, [[M_total]])
            write_matrix_file(py_N_tmp_path, py_N_path, [[pop_size]])
            print(f"[DEBUG] Wrote py_M={M_total}, py_N={pop_size}", flush=True)
            
            print(f"[DEBUG] Writing py_LOWER.txt and py_UPPER.txt...", flush=True)
            if run_ub is not None and run_lb is not None:
                write_matrix_file(py_lower_bound_tmp_path, py_lower_bound_path, [run_lb.tolist()])
                write_matrix_file(py_upper_bound_tmp_path, py_upper_bound_path, [run_ub.tolist()])
            else:
                write_matrix_file(py_lower_bound_tmp_path, py_lower_bound_path, [lower_bound.tolist()])
                write_matrix_file(py_upper_bound_tmp_path, py_upper_bound_path, [upper_bound.tolist()])
            print(f"[DEBUG] Wrote py_LOWER and py_UPPER", flush=True)
            
            print(f"[DEBUG] Writing py_init_pop.txt (len={len(init_pop)})...", flush=True)
            write_matrix_file(py_init_pop_tmp_path, py_init_pop_path, init_pop)
            print(f"[DEBUG] Wrote py_init_pop.txt", flush=True)
            
            print(f"[DEBUG] Removing py_NEW_RUN.txt...", flush=True)
            os.remove(py_new_run_path)
            # Handshake for MATLAB polling
            try:
                write_matrix_file(py_run_ready_tmp_path, py_run_ready_path, [[1]])
            except Exception:
                try:
                    py_run_ready_tmp_path.write_text('1')
                    py_run_ready_tmp_path.replace(py_run_ready_path)
                except Exception:
                    pass
            print(f"[DEBUG] New run setup complete.", flush=True)
        elif matlab_repair_emb_path.exists() and \
                not py_EMB_path.exists() and \
                not py_OBJ_path.exists():
            """
            MOMO-aligned evaluation: 
            1. 如果使用标准化空间，先将潜向量还原到原始空间
            2. 使用Beam Search解码，优先选择有效分子
            3. 移除repair操作（这是MOMO成功的关键）
            
            关键差异：MOMO不做repair，直接解码评估
            """

            tb = time.time()
            raw_emb = torch.tensor(read_matrix_file(matlab_repair_emb_path), dtype=torch.float32)
            
            # ===== Step 1: 如果使用标准化空间，还原到原始空间进行解码 =====
            if LATENT_NORMALIZE_TO_UNIT:
                raw_emb_np = raw_emb.numpy()
                raw_emb_np = optimizer.denormalize_z(raw_emb_np)
                raw_emb = torch.tensor(raw_emb_np, dtype=torch.float32)

            repair_mode = REPAIR_MODE
            if repair_mode not in {'always', 'none'}:
                repair_mode = 'always'

            # ===== Step 2: 解码获取SMILES =====
            tb_decode = time.time()
            
            if BEAM_SEARCH_ENABLED:
                # MOMO风格：Beam Search解码，优先返回有效分子
                decode_results = optimizer.decode_with_validity_fallback(
                    raw_emb, 
                    num_top=BEAM_SEARCH_NUM_TOP, 
                    temperature=DECODE_TEMPERATURE
                )
                seqs = [r[1] for r in decode_results]
                mols = [r[0] for r in decode_results]
            else:
                # 原始argmax解码
                seqs = optimizer.emb_to_seq(raw_emb)
                mols = [Chem.MolFromSmiles(s) for s in seqs]
            
            t_decode = time.time() - tb_decode

            # ===== Step 3: 决定是否repair（MOMO不做repair） =====
            if repair_mode == 'none':
                # MOMO风格：不做repair，直接使用解码结果
                # 输出的py_EMB就是输入的raw_emb（或标准化还原后的）
                if LATENT_NORMALIZE_TO_UNIT:
                    # 写回标准化空间的向量给PlatEMO
                    py_EMB_np = optimizer.normalize_z(raw_emb.numpy())
                    py_EMB = torch.tensor(py_EMB_np, dtype=torch.float32)
                else:
                    py_EMB = raw_emb
                desc = f"Decode(no_repair, beam={BEAM_SEARCH_ENABLED}): {t_decode:.3f}s"
            else:
                # 保留repair选项（向后兼容）
                repair_seqs = []
                gen = None
                if REPAIR_DECODE_STRATEGY == 'sample':
                    gen = torch.Generator(device='cpu')
                    gen.manual_seed(int(random_state) % (2**31 - 1))

                for j in range(raw_emb.shape[0]):
                    seq = seqs[j] if BEAM_SEARCH_ENABLED else None
                    
                    if seq is None or Chem.MolFromSmiles(seq) is None:
                        # 尝试其他解码策略
                        attempts = max(int(REPAIR_DECODE_ATTEMPTS), 1)
                        for a in range(attempts):
                            cand = optimizer.emb_to_seq_with_strategy(
                                raw_emb[j:j+1],
                                strategy=REPAIR_DECODE_STRATEGY,
                                temperature=REPAIR_DECODE_TEMPERATURE,
                                top_k=REPAIR_DECODE_TOP_K,
                                generator=gen,
                            )[0]
                            mol = Chem.MolFromSmiles(cand)
                            if mol is None:
                                continue
                            if REPAIR_AVOID_EXACT_LEAD and current_lead_smiles is not None:
                                if Chem.MolToSmiles(mol) == current_lead_smiles:
                                    continue
                            seq = cand
                            break
                        if seq is None:
                            seq = optimizer.emb_to_seq(raw_emb[j:j+1])[0]
                    repair_seqs.append(seq)
                
                # Re-encode for repair
                py_EMB_list = []
                for seq in repair_seqs:
                    z = optimizer.seq_to_emb(seq).squeeze()
                    if LATENT_NORMALIZE_TO_UNIT:
                        z_np = optimizer.normalize_z(z.detach().cpu().numpy())
                        z = torch.tensor(z_np, dtype=torch.float32)
                    py_EMB_list.append(z)
                py_EMB = torch.stack(py_EMB_list)
                seqs = repair_seqs  # 更新seqs用于目标评估
                
                desc = f"Repair(beam={BEAM_SEARCH_ENABLED}): {time.time() - tb:.3f}s"

            # 统计有效性和多样性
            valid_count = 0
            unique_smiles = set()
            lead_exact_count = 0
            for seq in seqs:
                total += 1
                mol = Chem.MolFromSmiles(seq)
                if mol is None:
                    invalid += 1
                else:
                    valid_count += 1
                    canonical = Chem.MolToSmiles(mol)
                    unique_smiles.add(canonical)
                    if current_lead_smiles and canonical == current_lead_smiles:
                        lead_exact_count += 1
            
            # 累积多样性统计（不每次打印，在 run 结束时汇总）
            diversity_stats['unique_total'] += len(unique_smiles)
            diversity_stats['valid_total'] += valid_count
            diversity_stats['lead_copies_total'] += lead_exact_count
            diversity_stats['eval_count'] += 1

            # Collect objectives and prepare for writing back into file for matlab to read
            tb = time.time()
            if objective_mode in ('momo_task1', 'momo_taskb'):
                # Task1: QED + similarity (2目标)
                # obj1: minimize -QED (越小越好 => QED 越大越好)
                # obj2: minimize -sim (越小越好 => sim 越大越好)
                if current_lead_fp is None:
                    raise RuntimeError("objective_mode=momo_task1 requires current_lead_fp (lead must be set before evaluation)")
                qed_vals = []
                sim_vals = []
                qed_raw = []
                sim_raw = []
                for smi in seqs:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        qed_vals.append(1.0)   # 最差（因为 -QED 的有效范围在 [-1,0]）
                        sim_vals.append(0.0)   # 最差（-sim 的最差为 0）
                        qed_raw.append(0.0)
                        sim_raw.append(0.0)
                        continue
                    q = float(QED.qed(mol))
                    s = float(tanimoto_similarity(mol, current_lead_fp))
                    qed_raw.append(q)
                    sim_raw.append(s)
                    qed_vals.append(-q)
                    sim_vals.append(-s)

                # Optional: penalize duplicate SMILES within the evaluated batch.
                # This can help reduce population collapse (many-to-one repair) without
                # changing M or the overall pipeline.
                if DUPLICATE_SMILES_PENALTY > 0:
                    seen = {}
                    for idx, smi in enumerate(seqs):
                        if smi in seen:
                            qed_vals[idx] = float(qed_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            sim_vals[idx] = float(sim_vals[idx]) + DUPLICATE_SMILES_PENALTY
                        else:
                            seen[smi] = 1

                obj = torch.tensor(np.stack([qed_vals, sim_vals], axis=1), dtype=torch.float32)

                # Cache last evaluated batch (used as final-pop proxy for MOMO-style record)
                if current_lead_id is not None:
                    last_eval_smiles = list(seqs)
                    last_eval_qed_raw = list(qed_raw)
                    last_eval_sim_raw = list(sim_raw)
                    last_eval_mol_id = int(current_lead_id)
                    last_eval_objective_mode = objective_mode
            
            elif objective_mode == 'momo_task2':
                # Task2: pLogP + similarity (2目标)
                # obj1: minimize -pLogP (越小越好 => pLogP 越大越好)
                # obj2: minimize -sim (越小越好 => sim 越大越好)
                # 注意：MOMO Task2实际使用的是pLogP的原始值（不是improvement）
                # 在metric计算时，会用 lead_plogp - current_plogp 来计算 improvement
                if current_lead_fp is None:
                    raise RuntimeError("objective_mode=momo_task2 requires current_lead_fp (lead must be set before evaluation)")
                plogp_vals = []
                sim_vals = []
                for smi in seqs:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        plogp_vals.append(20.0)  # 最差（MOMO用20表示无效）
                        sim_vals.append(0.0)     # 最差
                        continue
                    plogp = penalized_logP(mol)
                    if np.isnan(plogp):
                        plogp = 20.0
                    plogp_vals.append(-plogp)  # minimize -pLogP => maximize pLogP
                    sim_vals.append(-float(tanimoto_similarity(mol, current_lead_fp)))

                # Optional: penalize duplicate SMILES
                if DUPLICATE_SMILES_PENALTY > 0:
                    seen = {}
                    for idx, smi in enumerate(seqs):
                        if smi in seen:
                            plogp_vals[idx] = float(plogp_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            sim_vals[idx] = float(sim_vals[idx]) + DUPLICATE_SMILES_PENALTY
                        else:
                            seen[smi] = 1

                obj = torch.tensor(np.stack([plogp_vals, sim_vals], axis=1), dtype=torch.float32)
            
            elif objective_mode == 'momo_task3':
                # Task3: QED + DRD2 + similarity (3目标)
                # obj1: minimize -QED (越小越好 => QED 越大越好)
                # obj2: minimize -DRD2 (越小越好 => DRD2 越大越好)
                # obj3: minimize -sim (越小越好 => sim 越大越好)
                # 成功条件: QED >= 0.8, DRD2 >= 0.3, sim >= 0.4
                if current_lead_fp is None:
                    raise RuntimeError("objective_mode=momo_task3 requires current_lead_fp (lead must be set before evaluation)")
                if drd2_oracle is None:
                    raise RuntimeError("objective_mode=momo_task3 requires TDC package. Install with: pip install PyTDC")
                
                qed_vals = []
                drd2_vals = []
                sim_vals = []
                drd2_raw_vals = []
                drd2_nan_count = 0
                for smi in seqs:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        qed_vals.append(1.0)   # 最差
                        drd2_vals.append(1.0)  # 最差
                        sim_vals.append(0.0)   # 最差
                        drd2_raw_vals.append(np.nan)
                        drd2_nan_count += 1
                        continue
                    qed_vals.append(-float(QED.qed(mol)))
                    # DRD2 oracle 接受 SMILES 字符串
                    try:
                        drd2_score = float(drd2_oracle(smi))
                    except Exception:
                        drd2_score = np.nan
                    if not np.isfinite(drd2_score):
                        drd2_score = 0.0
                        drd2_nan_count += 1
                    drd2_raw_vals.append(drd2_score)
                    drd2_vals.append(-float(drd2_score))
                    sim_vals.append(-float(tanimoto_similarity(mol, current_lead_fp)))

                # Optional: penalize duplicate SMILES
                if DUPLICATE_SMILES_PENALTY > 0:
                    seen = {}
                    for idx, smi in enumerate(seqs):
                        if smi in seen:
                            qed_vals[idx] = float(qed_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            drd2_vals[idx] = float(drd2_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            sim_vals[idx] = float(sim_vals[idx]) + DUPLICATE_SMILES_PENALTY
                        else:
                            seen[smi] = 1

                drd2_raw_arr = np.asarray(drd2_raw_vals, dtype=np.float32)
                if drd2_raw_arr.size > 0:
                    pos_cnt = int(np.sum(drd2_raw_arr > 0))
                    print(
                        "[TASK3 DRD2 DEBUG] "
                        f"n={drd2_raw_arr.size}, "
                        f"nan_or_invalid={drd2_nan_count}, "
                        f"positive={pos_cnt}, "
                        f"min={np.nanmin(drd2_raw_arr):.6g}, "
                        f"max={np.nanmax(drd2_raw_arr):.6g}, "
                        f"mean={np.nanmean(drd2_raw_arr):.6g}",
                        flush=True
                    )
                    if pos_cnt == 0:
                        sample_n = min(3, len(seqs))
                        sample_pairs = [(seqs[k], float(drd2_raw_arr[k])) for k in range(sample_n)]
                        print(f"[TASK3 DRD2 SAMPLE ZERO] {sample_pairs}", flush=True)
                    else:
                        top_idx = np.argsort(-drd2_raw_arr)[:3]
                        top_pairs = [(seqs[int(k)], float(drd2_raw_arr[int(k)])) for k in top_idx]
                        print(f"[TASK3 DRD2 TOP3] {top_pairs}", flush=True)
                    # Sentinel check: verify DRD2 oracle is alive in the same process/context.
                    sentinel_smi = "O=C(CCCc1nnc(Cc2c[nH]c3ccccc23)o1)NC1CCCC1"
                    try:
                        sentinel_score = float(drd2_oracle(sentinel_smi))
                    except Exception:
                        sentinel_score = np.nan
                    print(f"[TASK3 SENTINEL DRD2] {sentinel_score:.10f}", flush=True)

                obj = torch.tensor(np.stack([qed_vals, drd2_vals, sim_vals], axis=1), dtype=torch.float32)
            
            elif objective_mode == 'momo_task4':
                # Task4: QED + GSK3β + SA_norm + similarity (4目标)
                # obj1: minimize -QED (越小越好 => QED 越大越好)
                # obj2: minimize -GSK3β (越小越好 => GSK3β 越大越好)
                # obj3: minimize -SA_norm (越小越好 => SA_norm 越大越好，合成越容易)
                # obj4: minimize -sim (越小越好 => sim 越大越好)
                # 成功条件: QED >= 0.8, GSK3β >= 0.5, SA_norm >= 0.8, sim >= 0.3
                if current_lead_fp is None:
                    raise RuntimeError("objective_mode=momo_task4 requires current_lead_fp (lead must be set before evaluation)")
                if gsk3b_oracle is None or sa_oracle is None:
                    raise RuntimeError("objective_mode=momo_task4 requires TDC package. Install with: pip install PyTDC")
                
                qed_vals = []
                gsk3b_vals = []
                sa_vals = []
                sim_vals = []
                for smi in seqs:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        qed_vals.append(1.0)    # 最差
                        gsk3b_vals.append(1.0)  # 最差
                        sa_vals.append(1.0)     # 最差
                        sim_vals.append(0.0)    # 最差
                        continue
                    qed_vals.append(-float(QED.qed(mol)))
                    # GSK3β oracle 接受 SMILES 字符串
                    gsk3b_score = gsk3b_oracle(smi)
                    if np.isnan(gsk3b_score):
                        gsk3b_score = 0.0
                    gsk3b_vals.append(-float(gsk3b_score))
                    # SA normalized score
                    sa_norm = normalize_sa(smi)
                    if np.isnan(sa_norm):
                        sa_norm = 0.0
                    sa_vals.append(-float(sa_norm))
                    sim_vals.append(-float(tanimoto_similarity(mol, current_lead_fp)))

                # Optional: penalize duplicate SMILES
                if DUPLICATE_SMILES_PENALTY > 0:
                    seen = {}
                    for idx, smi in enumerate(seqs):
                        if smi in seen:
                            qed_vals[idx] = float(qed_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            gsk3b_vals[idx] = float(gsk3b_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            sa_vals[idx] = float(sa_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            sim_vals[idx] = float(sim_vals[idx]) + DUPLICATE_SMILES_PENALTY
                        else:
                            seen[smi] = 1

                obj = torch.tensor(np.stack([qed_vals, gsk3b_vals, sa_vals, sim_vals], axis=1), dtype=torch.float32)
            
            elif objective_mode == 'momo_task5':
                # Task5: Pioglitazone MPO (Guacamol benchmark) - 4目标
                # obj1: minimize -dissimilarity (越小越好 => 与Pioglitazone越不相似越好)
                # obj2: minimize -MW_score (越小越好 => MW越接近目标越好)
                # obj3: minimize -RB_score (越小越好 => 旋转键数越接近2越好)
                # obj4: minimize -sim (越小越好 => 与lead越相似越好)
                # 注意：这是一个de novo设计任务，目标是生成与Pioglitazone结构不同但性质相似的分子
                if current_lead_fp is None:
                    raise RuntimeError("objective_mode=momo_task5 requires current_lead_fp (lead must be set before evaluation)")
                
                dissim_vals = []
                mw_vals = []
                rb_vals = []
                sim_vals = []
                for smi in seqs:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        dissim_vals.append(1.0)  # 最差
                        mw_vals.append(1.0)      # 最差
                        rb_vals.append(1.0)      # 最差
                        sim_vals.append(0.0)     # 最差
                        continue
                    # Dissimilarity to Pioglitazone (higher = more dissimilar = better)
                    dissim_vals.append(-float(calc_pioglitazone_dissimilarity(mol)))
                    # MW score (closer to target = better)
                    mw_vals.append(-float(calc_mw_score(mol)))
                    # Rotatable bonds score (closer to 2 = better)
                    rb_vals.append(-float(calc_rotatable_bonds_score(mol)))
                    # Similarity to lead molecule
                    sim_vals.append(-float(tanimoto_similarity(mol, current_lead_fp)))

                # Optional: penalize duplicate SMILES
                if DUPLICATE_SMILES_PENALTY > 0:
                    seen = {}
                    for idx, smi in enumerate(seqs):
                        if smi in seen:
                            dissim_vals[idx] = float(dissim_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            mw_vals[idx] = float(mw_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            rb_vals[idx] = float(rb_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            sim_vals[idx] = float(sim_vals[idx]) + DUPLICATE_SMILES_PENALTY
                        else:
                            seen[smi] = 1

                obj = torch.tensor(np.stack([dissim_vals, mw_vals, rb_vals, sim_vals], axis=1), dtype=torch.float32)
            
            elif objective_mode == 'momo_task6':
                # Task6: QED + Docking + similarity (3目标)
                # obj1: minimize -QED (越小越好 => QED 越大越好)
                # obj2: minimize docking_score (越小越好，负值更好)
                # obj3: minimize -sim (越小越好 => sim 越大越好)
                # 成功条件: QED >= 0.8, docking <= -10, sim >= 0.3
                # 注意：docking score 本身就是越负越好，所以直接最小化
                if current_lead_fp is None:
                    raise RuntimeError("objective_mode=momo_task6 requires current_lead_fp (lead must be set before evaluation)")
                
                qed_vals = []
                docking_vals = []
                sim_vals = []
                for smi in seqs:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        qed_vals.append(1.0)      # 最差
                        docking_vals.append(1000.0)  # 最差 (大正值)
                        sim_vals.append(0.0)      # 最差
                        continue
                    qed_vals.append(-float(QED.qed(mol)))
                    # Docking score (lower/more negative = better)
                    # 直接使用原始 docking score，不取负（因为本身就是越负越好）
                    dock_score = calc_docking_score(smi)
                    docking_vals.append(float(dock_score))  # minimize this
                    sim_vals.append(-float(tanimoto_similarity(mol, current_lead_fp)))

                # Optional: penalize duplicate SMILES
                if DUPLICATE_SMILES_PENALTY > 0:
                    seen = {}
                    for idx, smi in enumerate(seqs):
                        if smi in seen:
                            qed_vals[idx] = float(qed_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            docking_vals[idx] = float(docking_vals[idx]) + DUPLICATE_SMILES_PENALTY
                            sim_vals[idx] = float(sim_vals[idx]) + DUPLICATE_SMILES_PENALTY
                        else:
                            seen[smi] = 1

                obj = torch.tensor(np.stack([qed_vals, docking_vals, sim_vals], axis=1), dtype=torch.float32)
            else:
                res = optimizer.properties(seqs)
                clf_res = res['clf'][:, clf_objectives]
                reg_res = res['reg'][:, reg_objectives]
                clf_res = convert_obj(clf_res, clf_objectives, clf_obj_type)
                reg_res = convert_obj(reg_res, reg_objectives, reg_obj_type)
                obj = torch.cat([clf_res, reg_res], dim=1)

                if 'SA_Score' in additional_objectives:
                    sa_obj = torch.tensor([sa_score(smi) for smi in seqs]).unsqueeze(dim=1)
                    obj = torch.cat([obj, sa_obj], dim=1)

                # Lead similarity objective (default still available in非MOMO模式)
                if current_lead_fp is not None:
                    sim_vals = []
                    for smi in seqs:
                        mol = Chem.MolFromSmiles(smi)
                        sim_vals.append(-float(tanimoto_similarity(mol, current_lead_fp)))
                    sim_obj = torch.tensor(sim_vals, dtype=torch.float32).unsqueeze(dim=1)
                    obj = torch.cat([obj, sim_obj], dim=1)

            # Final guardrail debug: inspect objective matrix right before writing py_OBJ.txt
            if objective_mode == 'momo_task3':
                obj_np = obj.detach().cpu().numpy()
                obj2 = obj_np[:, 1]            # stored objective: -DRD2
                drd2_back = -obj2              # recovered DRD2
                nz = int(np.sum(drd2_back > 0))
                print(
                    "[TASK3 PRE-WRITE] "
                    f"obj2(min,max,mean)=({obj2.min():.10g},{obj2.max():.10g},{obj2.mean():.10g}) "
                    f"drd2(min,max,mean)=({drd2_back.min():.10g},{drd2_back.max():.10g},{drd2_back.mean():.10g}) "
                    f"positive={nz}/{drd2_back.size}",
                    flush=True
                )
                sample_n = min(5, drd2_back.size)
                print(
                    f"[TASK3 PRE-WRITE SAMPLE DRD2] {drd2_back[:sample_n].tolist()}",
                    flush=True
                )

            obj = obj.tolist()

            write_matrix_file(py_OBJ_tmp_path, py_OBJ_path, obj)
            write_matrix_file(py_EMB_tmp_path, py_EMB_path, py_EMB.tolist())
            os.remove(matlab_repair_emb_path)

            # print(f"pyOBJ {time.time() * 1000}")  # used for debugging deadlock w/ MatLab
            t3 = time.time() - tb
            desc += f", py_OBJ: {t3}"

            pbar.update(1)
            pbar.set_description(desc)
