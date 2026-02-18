import os
import sys
import argparse
import json
import numpy as np
import torch
import re
from typing import List, Tuple

from config import MODEL_CONFIG, TRAIN_CONFIG
from utils import calculate_node_dim, get_device
from model import create_rgt_model

# Default thresholds
MSA_THRESHOLDS = {"both": 0.28, "single": 0.27, "evo": 0.28}

# Three-letter to one-letter amino-acid mapping (used to recover sequence from PDB)
AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extract the one-letter amino-acid sequence from ATOM records in a PDB file."""
    seq = ""
    current_pos = None
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                res_no = int(line[22:26].strip())
                res_3 = line[17:20].strip()
            except Exception:
                continue
            if current_pos is None or res_no != current_pos:
                current_pos = res_no
                seq += AA3_TO_1.get(res_3, "X")
    return seq


def write_results_txt(out_path: str, seq: str, scores: np.ndarray, threshold: float) -> None:
    probs = [round(float(s), 4) for s in scores]
    preds = [1 if p >= threshold else 0 for p in probs]
    header = f"The threshold of the predictive score to determine protein-DNA binding sites is set to {threshold}.\n"
    header += "AA\tProb\tPred\n"
    lines = [header]
    for i, aa in enumerate(seq):
        if i >= len(probs):
            break
        lines.append(f"{aa}\t{probs[i]}\t{preds[i]}\n")
    with open(out_path, "w") as f:
        f.writelines(lines)


def load_labels_from_fasta(fasta_path: str, protein_id: str) -> List[int]:
    """Read ground-truth labels of the specified protein from a FASTA file."""
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    with open(fasta_path, "r") as f:
        content = f.read()

    # Find the corresponding entry
    pattern = f">({protein_id})\\n([A-Z]+)\\n([01]+)"
    match = re.search(pattern, content)

    if not match:
        raise ValueError(f"Protein ID '{protein_id}' not found in FASTA file")

    found_id, sequence, labels_str = match.groups()
    labels = [int(c) for c in labels_str]

    return labels


def parse_pred_file(path: str) -> List[int]:
    """Parse a prediction TXT file and return predicted labels per residue."""
    preds: List[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and header lines
            if not line or line.startswith("The threshold") or line.startswith("AA\t"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 3:
                try:
                    preds.append(int(parts[2]))
                except ValueError:
                    pass
    return preds


def compute_counts(labels: List[int], preds: List[int]) -> Tuple[int, int, int, int, int]:
    """Compute TP, FP, FN, TN, and the aligned total length."""
    n = min(len(labels), len(preds))
    tp = fp = fn = tn = 0
    for p, l in zip(preds[:n], labels[:n]):
        if p == 1 and l == 1:
            tp += 1
        elif p == 1 and l == 0:
            fp += 1
        elif p == 0 and l == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn, n


def load_single_protein_features(data_path: str, protein_id: str, msa: str):
    feature_list = []

    # AF2 single-sequence representation
    if msa in ["both", "single"]:
        single_file = os.path.join(data_path, f"{protein_id}_single_norm.npy")
        if not os.path.exists(single_file):
            raise FileNotFoundError(f"Missing feature file: {single_file}")
        single = np.load(single_file)
        feature_list.append(single)

    # PSSM and HHM
    if msa in ["both", "evo"]:
        pssm_file = os.path.join(data_path, f"{protein_id}_pssm.npy")
        hhm_file = os.path.join(data_path, f"{protein_id}_hhm.npy")
        if not os.path.exists(pssm_file) or not os.path.exists(hhm_file):
            raise FileNotFoundError(f"Missing feature files: {pssm_file} or {hhm_file}")
        pssm = np.load(pssm_file)
        hhm = np.load(hhm_file)
        feature_list.extend([pssm, hhm])

    # DSSP
    dssp_file = os.path.join(data_path, f"{protein_id}_dssp.npy")
    if not os.path.exists(dssp_file):
        raise FileNotFoundError(f"Missing feature file: {dssp_file}")
    dssp = np.load(dssp_file)
    feature_list.append(dssp)

    # Distance map
    dismap_file = os.path.join(data_path, f"{protein_id}_dismap.npy")
    if not os.path.exists(dismap_file):
        raise FileNotFoundError(f"Missing distance map: {dismap_file}")
    distance_map = np.load(dismap_file)

    # Stack node features
    node_features = np.hstack(feature_list)
    return node_features, distance_map


def build_model_ensemble(device: torch.device, model_dir: str, msa: str):
    """Build a cross-validation model ensemble and load weights."""
    models = []
    model_cfg = MODEL_CONFIG.copy()
    model_cfg["protein_in_dim"] = calculate_node_dim(msa)

    # Build K-fold models
    for fold in range(TRAIN_CONFIG['num_folds']):
        model = create_rgt_model(device, model_cfg)
        ckpt = os.path.join(model_dir, f"fold{fold}.ckpt")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def predict_one_protein(data_path: str, model_dir: str, protein_id: str, msa: str) -> np.ndarray:
    """Run ensemble prediction for a single protein and return per-residue probabilities."""
    device = get_device()

    # Load features
    node_np, dist_np = load_single_protein_features(data_path, protein_id, msa)
    L, F = node_np.shape
    node_features = torch.from_numpy(node_np).float().unsqueeze(0).to(device)
    distance_map = torch.from_numpy(dist_np).float().unsqueeze(0).to(device)
    masks = torch.ones(1, L, dtype=torch.float32, device=device)

    # Models
    models = build_model_ensemble(device, model_dir, msa)

    # Predict
    with torch.no_grad():
        fold_preds = []
        for model in models:
            outputs = model(node_features, None, distance_map, masks)
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0, :L]
            fold_preds.append(probs)
    return np.mean(fold_preds, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Predict per-residue binding probabilities for a single protein and compute metrics"
    )
    parser.add_argument("--path", type=str, default="6ymw_B",
                        help="Directory containing input files (PDB and extracted features)")
    parser.add_argument("--id", type=str, default="6ymw_B",
                        help="Protein ID (basename of files like <ID>.pdb)")
    parser.add_argument("--msa", type=str, default="both", choices=["both", "single", "evo"],
                        help="MSA information type used by the model")
    parser.add_argument("--model_path", type=str, default='./newmodelresults/both/train/20250906_153405',
                        help="Path to model directory containing fold*.ckpt")
    parser.add_argument("--threshold", type=float, default=0.39,
                        help="Decision threshold; defaults to preset by msa type")
    parser.add_argument("--output", type=str, default='./6ymw_B/my6ymw_B_both_results.txt',
                        help="Output TXT path; defaults to <path>/<id>_<msa>_results.txt")
    parser.add_argument("--fasta_path", type=str, default='./Dataset/DNA_Test_181.fa',
                        help="Path to FASTA file containing ground truth labels")
    args = parser.parse_args()

    data_path = args.path if args.path.endswith(os.sep) else args.path + os.sep
    protein_id = args.id
    msa = args.msa
    model_dir = args.model_path

    # Parse threshold
    threshold = float(args.threshold if args.threshold is not None else MSA_THRESHOLDS.get(msa, 0.28))

    # Validate inputs
    pdb_file = os.path.join(data_path, f"{protein_id}.pdb")
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB not found: {pdb_file}")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    seq = extract_sequence_from_pdb(pdb_file)

    # Run prediction
    scores = predict_one_protein(data_path, model_dir, protein_id, msa)

    # Write outputs
    out_path = args.output or os.path.join(data_path, f"{protein_id}_{msa}_results.txt")
    write_results_txt(out_path, seq, scores, threshold)
    print(f"Results are saved in {out_path}")

    # Load ground-truth labels from FASTA
    labels = load_labels_from_fasta(args.fasta_path, protein_id)

    # Parse predicted labels from output file
    preds = parse_pred_file(out_path)

    # Compute counts and metrics
    tp, fp, fn, tn, n = compute_counts(labels, preds)

    # Print results
    length_note = "" if len(preds) == len(labels) else (
        f" (aligned min length: {min(len(preds), len(labels))}, labels={len(labels)}, preds={len(preds)})"
    )
    print(f"\n=== Metrics for {protein_id} ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn} N={n}{length_note}")


if __name__ == "__main__":
    main()
