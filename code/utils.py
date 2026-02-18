import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve
from scipy import stats
import pickle
import pandas as pd
import csv
import datetime
from config import *
import logging
import sys


def seed_everything(seed=2021):
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def find_optimal_threshold(y_true, y_prob):
    """Find the optimal threshold by maximizing the F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]

    return optimal_threshold, best_f1


def calculate_metrics_with_threshold(y_true, y_prob, threshold):
    """Compute evaluation metrics using a specified threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    # Validate inputs
    if len(y_true) == 0 or len(y_prob) == 0:
        return get_nan_metrics(threshold)

    try:
        # Use standard sklearn metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

        # Compute core metrics directly
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        # Confusion matrix for TP, TN, FP, FN and specificity
        cm = confusion_matrix(y_true, y_pred)

        # Handle confusion matrix to ensure a 2x2 form
        if cm.size == 4:  # 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:  # 1x1 matrix: only one class exists
            if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
                if y_true[0] == 0 and y_pred[0] == 0:  # all negatives
                    tn, fp, fn, tp = len(y_true), 0, 0, 0
                elif y_true[0] == 1 and y_pred[0] == 1:  # all positives
                    tn, fp, fn, tp = 0, 0, 0, len(y_true)
                else:  # labels and predictions do not match
                    tn, fp, fn, tp = 0, 0, 0, 0
            else:
                # Rare abnormal cases
                tn, fp, fn, tp = 0, 0, 0, 0
        else:
            # Other cases: attempt to reconstruct as 2x2
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)

            if len(unique_true) == 1 and len(unique_pred) == 1:
                if unique_true[0] == 0 and unique_pred[0] == 0:
                    tn, fp, fn, tp = len(y_true), 0, 0, 0
                elif unique_true[0] == 1 and unique_pred[0] == 1:
                    tn, fp, fn, tp = 0, 0, 0, len(y_true)
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
            else:
                # Compute via standard boolean counts
                tp = np.sum((y_true == 1) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))

        # Specificity
        specificity = safe_divide(tn, tn + fp)

        # AUC and AUPR
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5  # default if fails

        try:
            aupr = average_precision_score(y_true, y_prob)
        except ValueError:
            aupr = np.mean(y_true)  # default to prevalence

        metrics = {
            'Spe': specificity,
            'Rec': recall,
            'Pre': precision,
            'F1': f1,
            'MCC': mcc,
            'AUC': auc,
            'AUPR': aupr,
            'threshold': threshold,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }

        return metrics

    except Exception as e:
        if not hasattr(calculate_metrics_with_threshold, '_printed_error'):
            print(f"Warning: Error calculating metrics: {e}")
            calculate_metrics_with_threshold._printed_error = True
        return get_nan_metrics(threshold)


def safe_divide(numerator, denominator, default=0.0):
    """Safe division that handles division-by-zero and NaN/Inf."""
    if denominator == 0:
        return default
    result = numerator / denominator
    if np.isnan(result) or np.isinf(result):
        return default
    return result


def get_nan_metrics(threshold):
    """Return a default metrics dict for exceptional cases."""
    return {
        'Spe': 0.0,
        'Rec': 0.0,
        'Pre': 0.0,
        'F1': 0.0,
        'MCC': 0.0,
        'AUC': 0.5,
        'AUPR': 0.0,
        'threshold': threshold,
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }


def calculate_metrics(y_true, y_pred=None, y_prob=None, threshold=None, use_fixed_threshold=False, msa_type='both'):
    """Compute evaluation metrics with an optional threshold or a default threshold policy."""
    if y_prob is None:
        return get_nan_metrics(threshold if threshold is not None else 0.5)

    if threshold is None:
        if use_fixed_threshold:
            # Use the default threshold from configuration
            threshold = get_default_threshold(msa_type)
        else:
            # Determine threshold by maximizing F1-score
            optimal_threshold, _ = find_optimal_threshold(y_true, y_prob)
            threshold = optimal_threshold

    return calculate_metrics_with_threshold(y_true, y_prob, threshold)


def save_results_to_csv(results_dict, csv_path, mode='a'):
    """Save results to a CSV file."""
    clean_results = {}
    for key, value in results_dict.items():
        if isinstance(value, (int, float, str, bool)):
            clean_results[key] = value
        elif isinstance(value, np.ndarray):
            clean_results[key] = str(value.tolist()) if value.size <= 10 else f"array_shape_{value.shape}"
        elif value is None:
            clean_results[key] = 'None'
        else:
            clean_results[key] = str(value)

    # If the file exists, read it; otherwise start with an empty table
    existing_df = None
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            # Retry using Python engine
            try:
                existing_df = pd.read_csv(csv_path, engine='python')
            except Exception:
                # Give up the old file if it is unreadable
                existing_df = None

    new_row_df = pd.DataFrame([clean_results])

    if existing_df is None or len(existing_df) == 0:
        # Write a new file
        new_row_df.to_csv(csv_path, index=False)
        return

    # Compute the union of columns and align
    all_columns = list(dict.fromkeys(list(existing_df.columns) + list(new_row_df.columns)))
    existing_df = existing_df.reindex(columns=all_columns)
    new_row_df = new_row_df.reindex(columns=all_columns)

    # Append and overwrite the whole file
    combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    combined_df.to_csv(csv_path, index=False)


def load_results_from_csv(csv_path):
    """Load results from CSV (compatible with older irregular files)."""
    if not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        try:
            return pd.read_csv(csv_path, engine='python')
        except Exception:
            return None


def determine_cv_threshold(cv_predictions_list, cv_labels_list):
    """Determine the optimal threshold from cross-validation results."""
    # Merge predictions and labels across folds
    all_predictions = np.concatenate(cv_predictions_list)
    all_labels = np.concatenate(cv_labels_list)

    # Compute optimal threshold
    optimal_threshold, best_f1 = find_optimal_threshold(all_labels, all_predictions)

    if not hasattr(determine_cv_threshold, '_printed_threshold'):
        print(f"Optimal threshold determined from CV: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
        determine_cv_threshold._printed_threshold = True

    return optimal_threshold


def save_threshold_to_csv(threshold, csv_path, msa_type):
    """Save a threshold value to CSV."""
    threshold_data = {
        'msa_type': msa_type,
        'optimal_threshold': threshold,
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    save_results_to_csv(threshold_data, csv_path, mode='w')


def load_threshold_from_csv(csv_path, msa_type):
    """Load a threshold value from CSV."""
    df = load_results_from_csv(csv_path)
    if df is not None:
        row = df[df['msa_type'] == msa_type]
        if not row.empty:
            return float(row['optimal_threshold'].iloc[0])
    return None


def calculate_per_protein_metrics(predictions_list, labels_list, threshold=None, use_fixed_threshold=False, msa_type='both'):
    """Compute per-protein metrics."""
    per_protein_metrics = []

    for pred, label in zip(predictions_list, labels_list):
        # Skip invalid data
        if len(pred) == 0 or len(label) == 0:
            continue

        # Check if it is a single-class label set
        if len(np.unique(label)) == 1:
            # Still compute metrics for single-class cases
            pass

        if threshold is not None:
            metrics = calculate_metrics_with_threshold(label, pred, threshold)
        else:
            metrics = calculate_metrics(label, None, pred, None, use_fixed_threshold, msa_type)
        per_protein_metrics.append(metrics)

    return per_protein_metrics


def calculate_confidence_accuracy(predictions, labels, uncertainty, confidence_levels=[0.05, 0.1, 0.15, 0.2]):
    """Compute metrics under different confidence levels."""
    confidence_accuracy = {}

    for conf_level in confidence_levels:
        confident_mask = uncertainty < conf_level
        if np.sum(confident_mask) > 0:
            conf_preds = predictions[confident_mask]
            conf_labels = labels[confident_mask]

            if len(np.unique(conf_labels)) > 1:
                metrics = calculate_metrics(conf_labels, None, conf_preds)
                confidence_accuracy[conf_level] = {
                    'n_residues': np.sum(confident_mask),
                    'aupr': metrics['AUPR'],
                    'auc': metrics['AUC'],
                    'mcc': metrics['MCC']
                }

    return confidence_accuracy


def save_fold_results_to_csv(fold_metrics_list, csv_path):
    """Save per-fold results to CSV."""
    all_results = []

    for fold_idx, metrics in enumerate(fold_metrics_list):
        result = {'fold': fold_idx}
        result.update(metrics)
        all_results.append(result)

    # Compute mean and std
    avg_result = {'fold': 'average'}
    std_result = {'fold': 'std'}

    for metric in METRICS:
        values = [fm[metric] for fm in fold_metrics_list]
        avg_result[metric] = np.mean(values)
        std_result[metric] = np.std(values)

    all_results.append(avg_result)
    all_results.append(std_result)

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)


def normalize_features(features, min_vals, max_vals):
    """Feature normalization."""
    return (features - min_vals) / (max_vals - min_vals + 1e-8)


def load_fasta_sequences(fasta_file):
    """Load sequences from a FASTA file."""
    sequences = {}
    current_id = None
    current_seq = ""

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = current_seq
                current_id = line[1:].split()[0]  # Use the first token after '>' as the ID
                current_seq = ""
            else:
                current_seq += line

    if current_id is not None:
        sequences[current_id] = current_seq

    return sequences


def parse_binding_labels(fasta_file):
    """Parse binding-site labels from a FASTA-like file."""
    binding_labels = {}

    with open(fasta_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            protein_id = lines[i][1:].strip().split()[0]
            i += 1
            sequence = ""
            labels = ""

            # Read sequence and labels
            while i < len(lines) and not lines[i].startswith('>'):
                line = lines[i].strip()
                if line.isdigit() or all(c in '01' for c in line):
                    labels += line
                else:
                    sequence += line
                i += 1

            # Ensure length consistency
            if len(sequence) != len(labels):
                if not hasattr(parse_binding_labels, '_printed_warnings'):
                    print(f"Warning: Length mismatch for {protein_id}: sequence={len(sequence)}, labels={len(labels)}")
                    parse_binding_labels._printed_warnings = True
                # Truncate to the shorter length
                min_len = min(len(sequence), len(labels))
                sequence = sequence[:min_len]
                labels = labels[:min_len]

            # Convert labels to numeric array
            if labels:
                label_array = np.array([int(x) for x in labels])
                binding_labels[protein_id] = {
                    'sequence': sequence,
                    'labels': label_array
                }
            else:
                if not hasattr(parse_binding_labels, '_printed_no_labels'):
                    print(f"Warning: No labels found for {protein_id}")
                    parse_binding_labels._printed_no_labels = True
        else:
            i += 1

    return binding_labels


def split_dataset_k_fold(protein_ids, k=5, seed=2021):
    """K-fold split for protein IDs."""
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(protein_ids)
    fold_size = len(shuffled_ids) // k

    folds = []
    for i in range(k):
        start_idx = i * fold_size
        if i == k - 1:  # the last fold contains remaining data
            end_idx = len(shuffled_ids)
        else:
            end_idx = (i + 1) * fold_size

        test_ids = shuffled_ids[start_idx:end_idx]
        train_ids = np.concatenate([shuffled_ids[:start_idx], shuffled_ids[end_idx:]])

        folds.append({'train': train_ids.tolist(), 'test': test_ids.tolist()})

    return folds


def random_sampling_with_replacement(data_list, n_samples, seed=None):
    """Random sampling with replacement."""
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.choice(len(data_list), size=n_samples, replace=True)
    sampled_data = [data_list[i] for i in indices]

    return sampled_data


class EarlyStopping:
    """Early stopping mechanism."""
    def __init__(self, patience=4, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format seconds into a human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.0f}s"


def print_metrics_table(metrics_dict, title="Results"):
    """Print a metrics table."""
    print(f"\n{title}")
    print("=" * 80)
    print(f"{'Metric':<10} {'Value':<10}")
    print("-" * 25)

    for metric, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{metric:<10} {value:<10.4f}")
        else:
            print(f"{metric:<10} {value:<10}")
    print("=" * 80)


def check_file_exists(filepath, description=""):
    """Check whether a file exists."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{description} file not found: {filepath}")
    return True


def get_device():
    """Get the compute device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def calculate_node_dim(msa_type):
    """Compute node feature dimension."""
    if msa_type == "both":
        return FEATURE_DIMS['single'] + FEATURE_DIMS['pssm'] + FEATURE_DIMS['hhm'] + FEATURE_DIMS['dssp']
    elif msa_type == "single":
        return FEATURE_DIMS['single'] + FEATURE_DIMS['dssp']
    elif msa_type == "evo":
        return FEATURE_DIMS['pssm'] + FEATURE_DIMS['hhm'] + FEATURE_DIMS['dssp']
    else:
        raise ValueError(f"Unknown msa_type: {msa_type}")


def setup_logger(logger_name, log_file, level=logging.INFO):
    """Set up a logger."""
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
