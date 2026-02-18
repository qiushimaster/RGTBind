import os
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

def get_required_files(protein_id):
    """返回指定蛋白ID所需特征文件名列表"""
    required_files = [
        f"{protein_id}_dssp.npy",      # DSSP特征
        f"{protein_id}_dismap.npy",    # 距离图特征
        f"{protein_id}_single_norm.npy", # AF2单表示
        f"{protein_id}_pssm.npy",      # PSSM特征
        f"{protein_id}_hhm.npy",       # HMM特征
        f"{protein_id}.fa",            # FASTA
        f"{protein_id}.dssp",          # DSSP原始
        f"{protein_id}.map",           # 距离图原始
        f"{protein_id}.bla",           # BLAST输出
        f"{protein_id}.pssm",          # PSSM原始
        f"{protein_id}.hhm",           # HMM原始
        f"{protein_id}.a3m",           # A3M
        f"{protein_id}.hhr"            # HHR
    ]
    return required_files

def check_protein_features(protein_dir, protein_id):
    """检查单个蛋白的特征文件是否齐全"""
    if not os.path.exists(protein_dir):
        return False, [], "目录不存在"
    
    required_files = get_required_files(protein_id)
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(protein_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    is_complete = len(missing_files) == 0
    return is_complete, missing_files, None

def check_dataset_features(feature_root, dataset_name):
    """检查指定数据集内各蛋白特征是否齐全"""
    dataset_path = os.path.join(feature_root, dataset_name)
    
    if not os.path.exists(dataset_path):
        # 仅首次输出详细信息
        if not hasattr(check_dataset_features, '_printed_path_error'):
            print(f"数据集路径不存在: {dataset_path}")
            check_dataset_features._printed_path_error = True
        return [], []
    
    # 仅首次打印检查信息
    if not hasattr(check_dataset_features, '_printed_headers'):
        print(f"\n检查数据集: {dataset_name}")
        print(f"路径: {dataset_path}")
        check_dataset_features._printed_headers = True
    
    # 获取所有蛋白文件夹
    protein_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            protein_dirs.append(item)
    
    if not hasattr(check_dataset_features, '_printed_count'):
        print(f"发现 {len(protein_dirs)} 个蛋白质文件夹")
        check_dataset_features._printed_count = True
    
    complete_proteins = []
    incomplete_proteins = []
    
    for protein_id in protein_dirs:
        protein_dir = os.path.join(dataset_path, protein_id)
        is_complete, missing_files, error = check_protein_features(protein_dir, protein_id)
        
        if is_complete:
            complete_proteins.append(protein_id)
        else:
            incomplete_proteins.append((protein_id, missing_files, error))
    
    # 打印统计信息
    print(f"\n统计结果:")
    print(f"  完整特征: {len(complete_proteins)} 个蛋白质")
    print(f"  缺失特征: {len(incomplete_proteins)} 个蛋白质")
    print(f"  总计: {len(protein_dirs)} 个蛋白质")
    
    if incomplete_proteins:
        incomplete_ids = [protein_info[0] for protein_info in incomplete_proteins]
        print(f"\n缺失特征的蛋白质 ({len(incomplete_ids)} 个):")
        print(f"  {', '.join(incomplete_ids)}")
    
    return complete_proteins, incomplete_proteins

def check_all_datasets(feature_root):
    """检查根目录下所有数据集的特征是否齐全"""
    if not os.path.exists(feature_root):
        # 仅首次输出详细信息
        if not hasattr(check_all_datasets, '_printed_root_error'):
            print(f"特征根目录不存在: {feature_root}")
            check_all_datasets._printed_root_error = True
        return
    
    print(f"检查特征根目录: {feature_root}")
    
    # 获取所有数据集
    datasets = []
    for item in os.listdir(feature_root):
        item_path = os.path.join(feature_root, item)
        if os.path.isdir(item_path):
            datasets.append(item)
    
    if not datasets:
        print("未发现任何数据集")
        return
    
    print(f"发现数据集: {', '.join(datasets)}")
    
    all_results = {}
    total_complete = 0
    total_incomplete = 0
    
    for dataset_name in datasets:
        complete, incomplete = check_dataset_features(feature_root, dataset_name)
        all_results[dataset_name] = {
            'complete': complete,
            'incomplete': incomplete
        }
        total_complete += len(complete)
        total_incomplete += len(incomplete)
    
    # 总体统计
    print(f"\n{'='*60}")
    print(f"总体统计:")
    print(f"  完整特征: {total_complete} 个蛋白质")
    print(f"  缺失特征: {total_incomplete} 个蛋白质")
    print(f"  总计: {total_complete + total_incomplete} 个蛋白质")
    print(f"{'='*60}")
    
    return all_results

def delete_incomplete_proteins(feature_root, dataset_name, incomplete_proteins):
    """删除特征文件不齐的蛋白文件夹"""
    if not incomplete_proteins:
        print(f"数据集 {dataset_name} 没有不完整的蛋白质")
        return
    
    print(f"\n删除数据集 {dataset_name} 中不完整的蛋白质文件夹...")
    
    deleted_count = 0
    for protein_info in incomplete_proteins:
        protein_id = protein_info[0]
        protein_dir = os.path.join(feature_root, dataset_name, protein_id)
        
        try:
            if os.path.exists(protein_dir):
                shutil.rmtree(protein_dir)
                print(f"  已删除: {protein_id}")
                deleted_count += 1
        except Exception as e:
            print(f"  删除失败 {protein_id}: {e}")
    
    print(f"成功删除 {deleted_count} 个不完整的蛋白质文件夹")


def main():
    parser = argparse.ArgumentParser(description="检查特征文件情况")
    parser.add_argument("--feature-root", type=str, default="Dataset/feature",
                       help="特征文件根目录 默认Dataset/feature")
    parser.add_argument("--dataset", type=str, 
                       choices=["Train_573", "Test_129", "Test_181"],
                       help="指定要检查的数据集 默认检查所有")
    parser.add_argument("--fix", type=bool, default=False,
                       help="删除不完整的蛋白质文件夹")
    
    args = parser.parse_args()
    
    # 检查特征根目录
    if not os.path.exists(args.feature_root):
        print(f"错误: 特征根目录不存在: {args.feature_root}")
        return 1
    
    # 按参数检查单个或全部
    if args.dataset:
        # 单个数据集
        complete, incomplete = check_dataset_features(args.feature_root, args.dataset)
        
        if args.fix and incomplete:
            # 删除不完整的蛋白质
            delete_incomplete_proteins(args.feature_root, args.dataset, incomplete)
    else:
        # 所有数据集
        all_results = check_all_datasets(args.feature_root)
        
        if args.fix:
            # 删除不完整的蛋白文件夹
            for dataset_name, results in all_results.items():
                if results['incomplete']:
                    print(f"\n处理数据集: {dataset_name}")
                    delete_incomplete_proteins(args.feature_root, dataset_name, results['incomplete'])
    
    return 0

if __name__ == "__main__":
    exit(main())
