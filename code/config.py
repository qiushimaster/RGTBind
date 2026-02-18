import os
import datetime

MODEL_CONFIG = {
    # 基础结构
    "protein_in_dim": 438,
    "hidden_unit": 64,
    "fc_layer": 2,
    "self_atten_layer": 2,
    "attention_heads": 4,
    "num_neighbor": 30,
    "fc_dropout": 0.2,
    "attention_dropout": 0.0,
    "class_num": 1,


    # RBF距离编码
    "use_rbf_distance": True,  # 开关 True启用RBF False为解析形式

    "rbf_learnable_sigma": False,  # 是否学习sigma False更稳
    "rbf_sigma_factor": 1.0,  # sigma初始因子

    "rbf_short_min": 0.0,  # 短程范围
    "rbf_short_max": 8.0,
    "rbf_short_kernels": 6,
    "rbf_medium_min": 5.0,  # 中程范围
    "rbf_medium_max": 15.0,
    "rbf_medium_kernels": 8,
    "rbf_long_min": 10.0,  # 长程范围
    "rbf_long_max": 25.0,
    "rbf_long_kernels": 6,
    
    "rbf_scale_init": 0.5,  # RBF融合缩放初值

    # 邻居选择方法 KNN或可学习阈值
    "selection_method": "learnable_threshold",  # knn 或 learnable_threshold
    "threshold_target_sparsity": 0.3,
    "threshold_sparsity_weight": 1e-5,
}

# 测试配置 指定已训练模型路径
TEST_CONFIG = {
    # 'model_path': './newmodelresults/both/train/20250907_160257',
    'model_path': './models/Gate+RBF',
}

# 路径配置
PATHS = {
    'feature_root': './Dataset/feature',         # 特征数据路径
    'results_root': './modelresults/',        # 结果路径
    'af2_pdb_folder': './Dataset/AF2_predicted_pdb/',
    'af2_single_folder': './Dataset/AF2_single_representation/',
    'train_fa': './Dataset/DNA_Train_573.fa',
    'test_129_fa': './Dataset/DNA_Test_129.fa',
    'test_181_fa': './Dataset/DNA_Test_181.fa',
    'caldis_ca': './script/caldis_CA',
    'max_single_npy': './script/Max_alphafold_single.npy',
    'min_single_npy': './script/Min_alphafold_single.npy'
}

# 训练配置
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
    
    # 学习率调度器
    'lr_scheduler': 'cosine',
    'min_lr': 1e-7,  # 余弦退火的最小学习率

    # 自定义步长调度器
    'custom_step_epochs': [3, 6, 9],
    'custom_step_lrs': [1.5e-4, 8e-5, 3e-5],
    
    # 阈值配置
    'use_fixed_threshold': False,  # 是否使用固定阈值
    'default_threshold': 0.28,     # 默认阈值 在固定阈值时使用
}

# 不同MSA类型默认阈值
DEFAULT_THRESHOLDS = {
    'both': 0.28,
    'single': 0.27,
    'evo': 0.28
}


GRID_SEARCH_CONFIG = {
    'batch_size': [8],
    'learning_rate': [1e-3,2e-3,5e-4,3e-4],
    'weight_decay': [1e-5,1e-6],
    'dropout_rate': [0.25,0.2,0.15],
}

# 可学习阈值方法网格参数
LEARNABLE_THRESHOLD_GRID_SEARCH_CONFIG = {
    'threshold_target_sparsity': [0.2, 0.3],
    'threshold_sparsity_weight': [1e-5, 1e-4],
}

# KNN方法网格参数 保留原有配置
KNN_GRID_SEARCH_CONFIG = {
    **GRID_SEARCH_CONFIG,  # 继承通用参数
    # 'num_neighbor': [20, 30, 40],  # 邻居数
}

# RBF方法网格参数
RBF_GRID_SEARCH_CONFIG = {
    **GRID_SEARCH_CONFIG,  # 继承通用
    # 'rbf_short_kernels': [4, 6, 8],
    # 'rbf_medium_kernels': [6, 8, 10],
    # 'rbf_long_kernels': [4, 6, 8],
    # 'rbf_sigma_factor': [0.8, 1.0, 1.2],
    # 'rbf_learnable_sigma': [False, True],  # 是否学习sigma
    'rbf_scale_init': [0.5, 1.0, 0.3, 0.8],
}

# 余弦退火调度器网格参数
COSINE_GRID_SEARCH_CONFIG = {
    'min_lr': [1e-6, 1e-7],
}

# 步长调度器网格参数
STEP_GRID_SEARCH_CONFIG = {
    'step_gamma': [0.5],
    'step_size': [3],
}

# 特征维度
FEATURE_DIMS = {
    'single': 384,  # AF2单表示
    'pssm': 20,     # PSSM
    'hhm': 20,      # HMM
    'dssp': 14      # DSSP结构特征
}

# 数据集配置
DATASET_CONFIG = {
    'train_dataset': 'Train_573',
    'test_datasets': ['Test_129', 'Test_181'],
    'num_workers': 16
}




# 外部软件路径
SOFTWARE_PATHS = {
    'psiblast': '/root/autodl-tmp/GraphSite-master/ncbi-blast-2.17.0+/bin/psiblast',
    'hhblits': '/root/autodl-tmp/GraphSite-master/hh-suite/build/bin/hhblits',
    'dssp': '/root/miniconda3/envs/graphsite/bin/mkdssp',
    # 'uniref90': './database/u90/uniref90',
    # 'uniclust30': './database/u30/UniRef30_2023_02'
}

# 氨基酸相关常量
AA_CODES = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
            "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
AA_ABBR = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
AA_DICT = dict(zip(AA_CODES, AA_ABBR))

# 归一化常量
MAX_PSSM = [8, 9, 9, 9, 12, 10, 8, 8, 12, 9, 7, 9, 12, 10, 9, 8, 9, 13, 11, 8]
MIN_PSSM = [-12, -12, -13, -13, -12, -11, -12, -12, -12, -12, -12, -12, -12, -12, -13, -12, -12, -13, -11, -12]
MAX_HHM = [12303, 12666, 12575, 12045, 12421, 12301, 12561, 12088, 12241, 11779, 12921, 12198, 12640, 12414, 12021, 11692, 11673, 12649, 12645, 12291]
MIN_HHM = [0] * 20

# DSSP相关常量
SS_TYPES = "HBEGITSC"
RSA_STD = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
           185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

# 评估指标名称
METRICS = ['Spe', 'Rec', 'Pre', 'F1', 'MCC', 'AUC', 'AUPR']

def get_timestamp():
    """获取当前时间戳"""
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def create_timestamped_dir(base_path, msa_type, mode='train'):
    """创建带时间戳目录"""
    timestamp = get_timestamp()
    timestamped_dir = os.path.join(base_path, msa_type, mode, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir, timestamp

def get_grid_search_config(lr_scheduler='cosine', selection_method='knn'):
    """根据调度器与邻居选择方法获取网格配置"""
    base_config = {}
    
    # 按邻居选择方法确定基础配置
    if selection_method == 'learnable_threshold':
        base_config = LEARNABLE_THRESHOLD_GRID_SEARCH_CONFIG
    elif selection_method == 'knn':
        base_config = KNN_GRID_SEARCH_CONFIG
    else:
        base_config = GRID_SEARCH_CONFIG
    
    # 如启用RBF 合并RBF网格
    if MODEL_CONFIG.get('use_rbf_distance', False):
        base_config = {**base_config, **RBF_GRID_SEARCH_CONFIG}
    
    # 根据调度器添加额外参数
    if lr_scheduler == 'cosine':
        # 合并余弦调度器参数
        return {**base_config, **COSINE_GRID_SEARCH_CONFIG}
    elif lr_scheduler == 'step':
        # 合并步长调度器参数  
        return {**base_config, **STEP_GRID_SEARCH_CONFIG}
    elif lr_scheduler == 'custom_step':
        # custom_step不单独网格 仅返回基础超参
        return base_config
    else:
        raise ValueError(f"Unsupported scheduler type: {lr_scheduler}")

def get_default_threshold(msa_type):
    """获取指定MSA类型默认阈值"""
    return DEFAULT_THRESHOLDS.get(msa_type, 0.28)

def get_model_path(msa_type):
    """获取指定MSA类型模型路径"""
    if 'model_paths' in TEST_CONFIG and msa_type in TEST_CONFIG['model_paths']:
        return TEST_CONFIG['model_paths'][msa_type]
    elif 'model_path' in TEST_CONFIG:
        return TEST_CONFIG['model_path']
    else:
        raise ValueError(f"No model path specified for MSA type: {msa_type}. "
                        f"Please set TEST_CONFIG['model_path'] or TEST_CONFIG['model_paths']['{msa_type}'] in config.py")

def validate_model_config():
    """验证模型配置一致性"""
    cfg = MODEL_CONFIG
    
    # 验证邻居选择方法
    selection_method = cfg.get("selection_method", "knn")
    if selection_method not in ["knn", "learnable_threshold"]:
        raise ValueError(f"Invalid selection_method: {selection_method}. Must be 'knn' or 'learnable_threshold'")
    
    return True

def get_config_summary():
    """获取当前配置摘要"""
    cfg = MODEL_CONFIG
    summary = {
        "Architecture": {
            "Selection Method": cfg.get("selection_method", "knn"),
            "RBF Distance": cfg.get("use_rbf_distance", False),
        },
        "Neighbor Selection": {},
    }
    
    # 添加邻居选择参数
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
    """验证模型路径是否存在必要文件"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # 检查必要文件
    required_files = []
    
    # 检查各折模型文件
    for fold in range(TRAIN_CONFIG['num_folds']):
        required_files.append(f'fold{fold}.ckpt')
    
    # 检查结果与阈值文件
    csv_files = ['results.csv', 'optimal_threshold.csv']
    csv_exists = any(os.path.exists(os.path.join(model_path, f)) for f in csv_files)
    
    # 至少需要各折模型
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required model files in {model_path}: {missing_files}")
    
    # 若缺少CSV文件则警告 不阻止运行
    if not csv_exists:
        print(f"Warning: No CSV result files found in {model_path}. Will use default thresholds.")
    
    return True

# 若不存在则创建目录
def create_directories():
    """创建必要目录结构"""
    for path in [PATHS['feature_root'], PATHS['results_root']]:
        os.makedirs(path, exist_ok=True)
        
    # 为不同MSA类型创建子目录
    for msa_type in ['both', 'single', 'evo']:
        os.makedirs(os.path.join(PATHS['results_root'], msa_type), exist_ok=True)
        # 创建train与test子目录
        for mode in ['train', 'test']:
            os.makedirs(os.path.join(PATHS['results_root'], msa_type, mode), exist_ok=True)