import os
import datetime

MODEL_CONFIG = {
    # Base architecture
    "protein_in_dim": 438,
    "hidden_unit": 64,
    "fc_layer": 2,
    "self_atten_layer": 2,
    "attention_heads": 4,
    "num_neighbor": 30,
    "fc_dropout": 0.2,
    "attention_dropout": 0.0,
    "class_num": 1,

    # RBF distance encoding
    "use_rbf_distance": True,  # Switch: True enables RBF; False uses analytic form

    "rbf_learnable_sigma": False,  # Whether to learn sigma; False is usually more stable
    "rbf_sigma_factor": 1.0,  # Initial sigma scaling factor

    "rbf_short_min": 0.0,  # Short-range distance interval
    "rbf_short_max": 8.0,
    "rbf_short_kernels": 6,
    "rbf_medium_min": 5.0,  # Medium-range distance interval
    "rbf_medium_max": 15.0,
    "rbf_medium_kernels": 8,
    "rbf_long_min": 10.0,  # Long-range distance interval
    "rbf_long_max": 25.0,
    "rbf_long_kernels": 6,

    "rbf_scale_init": 0.5,  # Initial scaling value for RBF fusion

    # Neighbor selection method: KNN or learnable threshold
    "selection_method": "learnable_threshold",  # "knn" or "learnable_threshold"
    "threshold_target_sparsity": 0.3,
    "threshold_sparsity_weight": 1e-5,
}

# Test configuration: specify the path to a trained model
TEST_CONFIG = {
    # 'model_path': './newmodelresults/both/train/20250907_160257',
    'model_path': './newmodelresults/both/train/20250906_182824/gs_sch_cosine_bs_8_lr_0.001_wd_1e-05_dr_0.2_num_neighbor_20_rbf_scale_init_0.5_minlr_1e-07',
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 4,
    'learning_rate': 2e-3,
    'weight_decay': 1e-5,
    'beta1': 0.9,
    'beta2': 0.99,
    'eps': 1e-5,
    'dropout_rate': 0.25,
    'max_epochs': 15,
    'patience': 4,
    'samples_per_epoch': 5000,
    'num_folds': 5,
    'seed': 42,

    # Learning rate scheduler
    'lr_scheduler': 'cosine',
    'min_lr': 1e-7,  # Minimum learning rate for cosine annealing

    # Custom step scheduler
    'custom_step_epochs': [3, 6, 9],
    'custom_step_lrs': [1.5e-4, 8e-5, 3e-5],

    # Threshold configuration
    'use_fixed_threshold': False,  # Whether to use a fixed threshold
    'default_threshold': 0.28,      # Default threshold (used when fixed threshold is enabled)
}

# Default thresholds for different MSA types
DEFAULT_THRESHOLDS = {
    'both': 0.28,
    'single': 0.27,
    'evo': 0.28
}

GRID_SEARCH_CONFIG = {
    'batch_size': [8],
    'learning_rate': [1e-3, 2e-3, 5e-4, 3e-4],
    'weight_decay': [1e-5, 1e-6],
    'dropout_rate': [0.25, 0.2, 0.15],
}

# Grid parameters for the learnable-threshold method
LEARNABLE_THRESHOLD_GRID_SEARCH_CONFIG = {
    'threshold_target_sparsity': [0.2, 0.3],
    'threshold_sparsity_weight': [1e-5, 1e-4],
}

# Grid parameters for the KNN method (keep the original configuration)
KNN_GRID_SEARCH_CONFIG = {
    **GRID_SEARCH_CONFIG,  # Inherit common parameters
    # 'num_neighbor': [20, 30, 40],  # Number of neighbors
}

# Grid parameters for the RBF method
RBF_GRID_SEARCH_CONFIG = {
    **GRID_SEARCH_CONFIG,  # Inherit common parameters
    # 'rbf_short_kernels': [4, 6, 8],
    # 'rbf_medium_kernels': [6, 8, 10],
    # 'rbf_long_kernels': [4, 6, 8],
    # 'rbf_sigma_factor': [0.8, 1.0, 1.2],
    # 'rbf_learnable_sigma': [False, True],  # Whether to learn sigma
    'rbf_scale_init': [0.5, 1.0, 0.3, 0.8],
}

# Grid parameters for the cosine annealing scheduler
COSINE_GRID_SEARCH_CONFIG = {
    'min_lr': [1e-6, 1e-7],
}

# Grid parameters for the step scheduler
STEP_GRID_SEARCH_CONFIG = {
    'step_gamma': [0.5],
    'step_size': [3],
}

# Feature dimensions
FEATURE_DIMS = {
    'single': 384,  # AF2 single representation
    'pssm': 20,     # PSSM
    'hhm': 20,      # HMM
    'dssp': 14      # DSSP structural features
}

# Dataset configuration
DATASET_CONFIG = {
    'train_dataset': 'Train_573',
    'test_datasets': ['Test_129', 'Test_181'],
    'num_workers': 16
}

# Path configuration
PATHS = {
    'feature_root': './Dataset/feature',         # Feature data path
    'results_root': './newmodelresults/',        # Results path
    'af2_pdb_folder': './Dataset/AF2_predicted_pdb/',
    'af2_single_folder': './Dataset/AF2_single_representation/',
    'train_fa': './Dataset/DNA_Train_573.fa',
    'test_129_fa': './Dataset/DNA_Test_129.fa',
    'test_181_fa': './Dataset/DNA_Test_181.fa',
    'caldis_ca': './script/caldis_CA',
    'max_single_npy': './script/Max_alphafold_single.npy',
    'min_single_npy': './script/Min_alphafold_single.npy'
}

# External software paths
SOFTWARE_PATHS = {
    'psiblast': '/root/autodl-tmp/RGTBind-master/ncbi-blast-2.17.0+/bin/psiblast',
    'hhblits': '/root/autodl-tmp/RGTBind-master/hh-suite/build/bin/hhblits',
    'dssp': '/root/miniconda3/envs/RGTBind/bin/mkdssp',
    # 'uniref90': './database/u90/uniref90',
    # 'uniclust30': './database/u30/UniRef30_2023_02'
}

# Amino-acid related constants
AA_CODES = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
            "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
AA_ABBR = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
AA_DICT = dict(zip(AA_CODES, AA_ABBR))

# Normalization constants
MAX_PSSM = [8, 9, 9, 9, 12, 10, 8, 8, 12, 9, 7, 9, 12, 10, 9, 8, 9, 13, 11, 8]
MIN_PSSM = [-12, -12, -13, -13, -12, -11, -12, -12, -12, -12, -12, -12, -12, -12, -13, -12, -12, -13, -11, -12]
MAX_HHM = [12303, 12666, 12575, 12045, 12421, 12301, 12561, 12088, 12241, 11779, 12921, 12198, 12640, 12414, 12021, 11692, 11673, 12649, 12645, 12291]
MIN_HHM = [0] * 20

# DSSP-related constants
SS_TYPES = "HBEGITSC"
RSA_STD = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
           185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

# Metric names
METRICS = ['Spe', 'Rec', 'Pre', 'F1', 'MCC', 'AUC', 'AUPR']


def get_timestamp():
    """Get the current timestamp string."""
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def create_timestamped_dir(base_path, msa_type, mode='train'):
    """Create a timestamped directory."""
    timestamp = get_timestamp()
    timestamped_dir = os.path.join(base_path, msa_type, mode, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir, timestamp


def get_grid_search_config(lr_scheduler='cosine', selection_method='knn'):
    """Get grid-search configuration based on the scheduler and neighbor selection method."""
    base_config = {}

    # Determine the base configuration according to the neighbor selection method
    if selection_method == 'learnable_threshold':
        base_config = LEARNABLE_THRESHOLD_GRID_SEARCH_CONFIG
    elif selection_method == 'knn':
        base_config = KNN_GRID_SEARCH_CONFIG
    else:
        base_config = GRID_SEARCH_CONFIG

    # If RBF is enabled, merge RBF grid parameters
    if MODEL_CONFIG.get('use_rbf_distance', False):
        base_config = {**base_config, **RBF_GRID_SEARCH_CONFIG}

    # Add extra parameters according to the scheduler
    if lr_scheduler == 'cosine':
        # Merge cosine scheduler parameters
        return {**base_config, **COSINE_GRID_SEARCH_CONFIG}
    elif lr_scheduler == 'step':
        # Merge step scheduler parameters
        return {**base_config, **STEP_GRID_SEARCH_CONFIG}
    elif lr_scheduler == 'custom_step':
        # No separate grid for custom_step; only return base hyper-parameters
        return base_config
    else:
        raise ValueError(f"Unsupported scheduler type: {lr_scheduler}")


def get_default_threshold(msa_type):
    """Get the default threshold for the specified MSA type."""
    return DEFAULT_THRESHOLDS.get(msa_type, 0.28)


def get_model_path(msa_type):
    """Get the model path for the specified MSA type."""
    if 'model_paths' in TEST_CONFIG and msa_type in TEST_CONFIG['model_paths']:
        return TEST_CONFIG['model_paths'][msa_type]
    elif 'model_path' in TEST_CONFIG:
        return TEST_CONFIG['model_path']
    else:
        raise ValueError(
            f"No model path specified for MSA type: {msa_type}. "
            f"Please set TEST_CONFIG['model_path'] or TEST_CONFIG['model_paths']['{msa_type}'] in config.py"
        )


def validate_model_config():
    """Validate model configuration consistency."""
    cfg = MODEL_CONFIG

    # Validate neighbor selection method
    selection_method = cfg.get("selection_method", "knn")
    if selection_method not in ["knn", "learnable_threshold"]:
        raise ValueError(
            f"Invalid selection_method: {selection_method}. Must be 'knn' or 'learnable_threshold'"
        )

    return True


def get_config_summary():
    """Get a summary of the current configuration."""
    cfg = MODEL_CONFIG
    summary = {
        "Architecture": {
            "Selection Method": cfg.get("selection_method", "knn"),
            "RBF Distance": cfg.get("use_rbf_distance", False),
        },
        "Neighbor Selection": {},
    }

    # Add neighbor selection parameters
    if cfg.get("selection_method") == "knn":
        summary["Neighbor Selection"] = {
            "Method": "KNN",
            "Num Neighbors": cfg.get("num_neighbor", 30),
        }
    elif cfg.get("selection_method") == "learnable_threshold":
        summary["Neighbor Selection"] = {
            "Method": "Learnable Threshold",
            "Target Sparsity": cfg.get("threshold_target_sparsity", 0.2),
            "Sparsity Weight": cfg.get("threshold_sparsity_weight", 1e-4),
        }

    return summary


def validate_model_path(model_path, msa_type):
    """Validate whether the model path contains the required files."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Check required files
    required_files = []

    # Check checkpoint files for each fold
    for fold in range(TRAIN_CONFIG['num_folds']):
        required_files.append(f'fold{fold}.ckpt')

    # Check result and threshold CSV files
    csv_files = ['results.csv', 'optimal_threshold.csv']
    csv_exists = any(os.path.exists(os.path.join(model_path, f)) for f in csv_files)

    # At minimum, checkpoint files for each fold are required
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(f"Missing required model files in {model_path}: {missing_files}")

    # If CSV files are missing, warn but do not block execution
    if not csv_exists:
        print(f"Warning: No CSV result files found in {model_path}. Will use default thresholds.")

    return True


# Create directories if they do not exist
def create_directories():
    """Create required directory structure."""
    for path in [PATHS['feature_root'], PATHS['results_root']]:
        os.makedirs(path, exist_ok=True)

    # Create subdirectories for different MSA types
    for msa_type in ['both', 'single', 'evo']:
        os.makedirs(os.path.join(PATHS['results_root'], msa_type), exist_ok=True)
        # Create train and test subdirectories
        for mode in ['train', 'test']:
            os.makedirs(os.path.join(PATHS['results_root'], msa_type, mode), exist_ok=True)
