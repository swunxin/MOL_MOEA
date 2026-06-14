import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED

import build_vocab
from model import ReLSO


@dataclass
class ModelBundle:
    name: str
    model: ReLSO
    model_str2num: Dict[str, int]
    vocab_str2num: Dict[str, int]
    model_num2str: Dict[int, str]
    vocab_num2str: Dict[int, str]


def parse_args():
    p = argparse.ArgumentParser(description="Compare latent-space geometry of two ReLSO checkpoints.")
    p.add_argument("--old-ckpt", required=True, help="Path to old (baseline) checkpoint.")
    p.add_argument("--new-ckpt", required=True, help="Path to new (candidate) checkpoint.")
    p.add_argument("--dataset-file", default="allmolgen_198max_SMILES_SELFIES_tokenlen.csv")
    p.add_argument("--seq-col", default="smiles")
    p.add_argument("--vocab-file", default="selfies_vocab.txt")
    p.add_argument("--selfies", type=int, default=1, help="1 for SELFIES encoding, 0 for SMILES.")
    p.add_argument("--max-seq-len", type=int, default=200)
    p.add_argument("--sample-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps-list", default="0.05,0.10,0.20", help="Noise scales for smoothness test.")
    p.add_argument("--neighbors", type=int, default=6, help="Neighbors per molecule per eps.")
    p.add_argument("--output-json", default="reports/latent_diagnose_report.json")
    return p.parse_args()


def load_bundle(name: str, ckpt_path: str, vocab_file: str, device: torch.device) -> ModelBundle:
    vocab = build_vocab.load_vocab_from_file(vocab_file)
    model_str2num, vocab_str2num = build_vocab.get_single_encoder(vocab)
    model_num2str, vocab_num2str = build_vocab.get_decoders(model_str2num, vocab_str2num)
    model = ReLSO.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval().to(device)
    return ModelBundle(name, model, model_str2num, vocab_str2num, model_num2str, vocab_num2str)


def encode_seq(seq: str, bundle: ModelBundle, max_seq_len: int, use_selfies: bool, device: torch.device) -> torch.Tensor:
    if use_selfies:
        enc = build_vocab.selfies_encode_molecule(seq, bundle.model_str2num, bundle.vocab_str2num)
    else:
        enc = build_vocab.smiles_encode_molecule(seq, bundle.model_str2num, bundle.vocab_str2num)
    enc = [bundle.model_str2num["<GLOBAL>"]] + enc
    if len(enc) > max_seq_len:
        enc = enc[:max_seq_len]
    x = torch.tensor(enc, dtype=torch.long, device=device)
    x = F.pad(x, (0, max_seq_len - len(enc)), value=0)
    return x


def decode_tokens(tokens: List[int], bundle: ModelBundle, use_selfies: bool) -> str:
    if use_selfies:
        sf_txt = build_vocab.selfies_decode_molecule(tokens, bundle.model_num2str, bundle.vocab_num2str)
        try:
            return sf.decoder(sf_txt)
        except Exception:
            return ""
    return build_vocab.smiles_decode_molecule(tokens, bundle.model_num2str, bundle.vocab_num2str)


def decode_z(z: torch.Tensor, bundle: ModelBundle, use_selfies: bool) -> List[str]:
    with torch.no_grad():
        logits = bundle.model.decode(z).permute(0, 2, 1)
        tokens = torch.argmax(torch.softmax(logits, dim=2), dim=2).cpu().tolist()
    return [decode_tokens(t, bundle, use_selfies) for t in tokens]


def mol_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def sample_sequences(dataset_file: str, seq_col: str, sample_size: int, seed: int) -> List[str]:
    df = pd.read_csv(dataset_file, usecols=[seq_col]).dropna()
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    return df[seq_col].astype(str).tolist()


def latent_scale_stats(z: np.ndarray) -> Dict[str, float]:
    l2 = np.linalg.norm(z, axis=1)
    return {
        "z_mean_abs": float(np.mean(np.abs(z))),
        "z_std_mean_dim": float(np.mean(np.std(z, axis=0))),
        "z_l2_mean": float(np.mean(l2)),
        "z_l2_std": float(np.std(l2)),
    }


def pairwise_dist_mean(z: np.ndarray, seed: int, pairs: int = 4096) -> float:
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    if n < 2:
        return 0.0
    idx1 = rng.integers(0, n, size=pairs)
    idx2 = rng.integers(0, n, size=pairs)
    d = np.linalg.norm(z[idx1] - z[idx2], axis=1)
    return float(np.mean(d))


def evaluate_model(bundle: ModelBundle, seqs: List[str], max_seq_len: int, use_selfies: bool, device: torch.device,
                   eps_list: List[float], neighbors: int, seed: int) -> Dict:
    xs = []
    for s in seqs:
        xs.append(encode_seq(s, bundle, max_seq_len, use_selfies, device))
    x_batch = torch.stack(xs, dim=0)

    with torch.no_grad():
        z0, _ = bundle.model.encode(x_batch)
    z0_np = z0.detach().cpu().numpy()

    scale = latent_scale_stats(z0_np)
    scale["pairwise_dist_mean"] = pairwise_dist_mean(z0_np, seed)

    # 2) Cycle consistency: z0 -> decode -> re-encode = z1
    rec_seqs = decode_z(z0, bundle, use_selfies)
    z1_list = []
    valid_rec = 0
    exact_rec = 0
    for original, rec in zip(seqs, rec_seqs):
        mol_rec = Chem.MolFromSmiles(rec) if rec else None
        if mol_rec is not None:
            valid_rec += 1
            mol_org = Chem.MolFromSmiles(original)
            if mol_org is not None:
                if Chem.MolToSmiles(mol_org) == Chem.MolToSmiles(mol_rec):
                    exact_rec += 1
        x_rec = encode_seq(rec if rec else "C", bundle, max_seq_len, use_selfies, device)
        z1, _ = bundle.model.encode(x_rec.unsqueeze(0))
        z1_list.append(z1.squeeze(0))
    z1 = torch.stack(z1_list, dim=0)

    cos = F.cosine_similarity(z0, z1, dim=1).detach().cpu().numpy()
    l2_shift = torch.norm(z1 - z0, p=2, dim=1).detach().cpu().numpy()
    cycle = {
        "decode_valid_rate": float(valid_rec / len(seqs)),
        "decode_exact_rate": float(exact_rec / len(seqs)),
        "cycle_cosine_mean": float(np.mean(cos)),
        "cycle_l2_shift_mean": float(np.mean(l2_shift)),
    }

    # 3) Neighborhood smoothness
    rng = np.random.default_rng(seed + 17)
    z_std = float(np.std(z0_np)) + 1e-8
    smooth = {}
    base_mols = [Chem.MolFromSmiles(s) for s in rec_seqs]
    base_fps = [mol_fp(m) if m is not None else None for m in base_mols]

    for eps in eps_list:
        valid = 0
        uniq = set()
        sims = []
        qed_delta = []
        logp_delta = []
        total = len(seqs) * neighbors

        noise = rng.standard_normal(size=(len(seqs) * neighbors, z0_np.shape[1])).astype(np.float32)
        z_rep = np.repeat(z0_np, neighbors, axis=0)
        z_nb = z_rep + (eps * z_std) * noise
        z_nb_t = torch.tensor(z_nb, dtype=torch.float32, device=device)
        nb_seqs = decode_z(z_nb_t, bundle, use_selfies)

        for i, smi in enumerate(nb_seqs):
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            valid += 1
            can = Chem.MolToSmiles(mol)
            uniq.add(can)

            base_idx = i // neighbors
            bfp = base_fps[base_idx]
            bmol = base_mols[base_idx]
            if bfp is not None:
                sims.append(float(DataStructs.TanimotoSimilarity(bfp, mol_fp(mol))))
            if bmol is not None:
                qed_delta.append(abs(float(QED.qed(mol)) - float(QED.qed(bmol))))
                logp_delta.append(abs(float(Descriptors.MolLogP(mol)) - float(Descriptors.MolLogP(bmol))))

        smooth[f"eps_{eps:.3f}"] = {
            "valid_rate": float(valid / total) if total else 0.0,
            "unique_rate_over_valid": float(len(uniq) / max(valid, 1)),
            "mean_sim_to_base": float(np.mean(sims)) if sims else 0.0,
            "mean_abs_delta_qed": float(np.mean(qed_delta)) if qed_delta else 0.0,
            "mean_abs_delta_logp": float(np.mean(logp_delta)) if logp_delta else 0.0,
        }

    return {
        "latent_scale": scale,
        "cycle_consistency": cycle,
        "neighborhood_smoothness": smooth,
    }


def verdict(old_res: Dict, new_res: Dict, eps_list: List[float]) -> Dict[str, str]:
    old_c = old_res["cycle_consistency"]
    new_c = new_res["cycle_consistency"]

    score_old = 0
    score_new = 0

    # Better decode validity / exactness / cosine; lower l2 shift
    if old_c["decode_valid_rate"] >= new_c["decode_valid_rate"]:
        score_old += 1
    else:
        score_new += 1
    if old_c["decode_exact_rate"] >= new_c["decode_exact_rate"]:
        score_old += 1
    else:
        score_new += 1
    if old_c["cycle_cosine_mean"] >= new_c["cycle_cosine_mean"]:
        score_old += 1
    else:
        score_new += 1
    if old_c["cycle_l2_shift_mean"] <= new_c["cycle_l2_shift_mean"]:
        score_old += 1
    else:
        score_new += 1

    # Smoothness: higher valid/similarity, lower property jumps
    for eps in eps_list:
        k = f"eps_{eps:.3f}"
        o = old_res["neighborhood_smoothness"][k]
        n = new_res["neighborhood_smoothness"][k]
        if o["valid_rate"] >= n["valid_rate"]:
            score_old += 1
        else:
            score_new += 1
        if o["mean_sim_to_base"] >= n["mean_sim_to_base"]:
            score_old += 1
        else:
            score_new += 1
        if o["mean_abs_delta_qed"] <= n["mean_abs_delta_qed"]:
            score_old += 1
        else:
            score_new += 1
        if o["mean_abs_delta_logp"] <= n["mean_abs_delta_logp"]:
            score_old += 1
        else:
            score_new += 1

    summary = (
        "old checkpoint appears less distorted"
        if score_old > score_new
        else "new checkpoint appears less distorted"
        if score_new > score_old
        else "tie: no clear winner"
    )
    return {"summary": summary, "score_old": str(score_old), "score_new": str(score_new)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_selfies = bool(args.selfies)
    eps_list = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]

    seqs = sample_sequences(args.dataset_file, args.seq_col, args.sample_size, args.seed)
    old_bundle = load_bundle("old", args.old_ckpt, args.vocab_file, device)
    new_bundle = load_bundle("new", args.new_ckpt, args.vocab_file, device)

    old_res = evaluate_model(old_bundle, seqs, args.max_seq_len, use_selfies, device, eps_list, args.neighbors, args.seed)
    new_res = evaluate_model(new_bundle, seqs, args.max_seq_len, use_selfies, device, eps_list, args.neighbors, args.seed)
    end_verdict = verdict(old_res, new_res, eps_list)

    report = {
        "config": {
            "old_ckpt": args.old_ckpt,
            "new_ckpt": args.new_ckpt,
            "dataset_file": args.dataset_file,
            "sample_size": args.sample_size,
            "eps_list": eps_list,
            "neighbors": args.neighbors,
            "selfies": use_selfies,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
        },
        "old": old_res,
        "new": new_res,
        "verdict": end_verdict,
    }

    print("\n=== Latent Diagnose Verdict ===")
    print(f"Summary: {end_verdict['summary']}")
    print(f"Score(old/new): {end_verdict['score_old']} / {end_verdict['score_new']}")
    print("Cycle consistency (old):", old_res["cycle_consistency"])
    print("Cycle consistency (new):", new_res["cycle_consistency"])

    out_path = args.output_json
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved full report to: {out_path}")


if __name__ == "__main__":
    main()
