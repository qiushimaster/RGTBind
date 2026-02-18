import os
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def get_required_files(protein_id):
    """Return the list of required feature filenames for the given protein ID."""
    required_files = [
        f"{protein_id}_dssp.npy",        # DSSP features
        f"{protein_id}_dismap.npy",      # distance map features
        f"{protein_id}_single_norm.npy", # AF2 single representation
        f"{protein_id}_pssm.npy",        # PSSM features
        f"{protein_id}_hhm.npy",         # HMM features
        f"{protein_id}.fa",              # FASTA
        f"{protein_id}.dssp",            # raw DSSP
        f"{protein_id}.map",             # raw distance map
        f"{protein_id}.bla",             # BLAST output
        f"{protein_id}.pssm",            # raw PSSM
        f"{protein_id}.hhm",             # raw HMM
        f"{protein_id}.a3m",             # A3M
        f"{protein_id}.hhr"              # HHR
    ]
    return required_files


def check_protein_features(protein_dir, protein_id):
    """Check whether all feature files for a single protein are present."""
    if not os.path.exists(protein_dir):
        return False, [], "Directory does not exist"

    required_files = get_required_files(protein_id)
    missing_files = []

    for file_name in required_files:
        file_path = os.path.join(protein_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)

    is_complete = len(missing_files) == 0
    return is_complete, missing_files, None


def check_dataset_features(feature_root, dataset_name):
    """Check whether features are complete for each protein in the specified dataset."""
    dataset_path = os.path.join(feature_root, dataset_name)

    if not os.path.exists(dataset_path):
        # Print detailed info only once
        if not hasattr(check_dataset_features, "_printed_path_error"):
            print(f"Dataset path does not exist: {dataset_path}")
            check_dataset_features._printed_path_error = True
        return [], []

    # Print header only once
    if not hasattr(check_dataset_features, "_printed_headers"):
        print(f"\nChecking dataset: {dataset_name}")
        print(f"Path: {dataset_path}")
        check_dataset_features._printed_headers = True

    # Collect all protein folders
    protein_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            protein_dirs.append(item)

    if not hasattr(check_dataset_features, "_printed_count"):
        print(f"Found {len(protein_dirs)} protein folders")
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

    # Print summary
    print("\nSummary:")
    print(f"  Complete features: {len(complete_proteins)} proteins")
    print(f"  Missing features: {len(incomplete_proteins)} proteins")
    print(f"  Total: {len(protein_dirs)} proteins")

    if incomplete_proteins:
        incomplete_ids = [protein_info[0] for protein_info in incomplete_proteins]
        print(f"\nProteins with missing features ({len(incomplete_ids)}):")
        print(f"  {', '.join(incomplete_ids)}")

    return complete_proteins, incomplete_proteins


def check_all_datasets(feature_root):
    """Check feature completeness for all datasets under the root directory."""
    if not os.path.exists(feature_root):
        # Print detailed info only once
        if not hasattr(check_all_datasets, "_printed_root_error"):
            print(f"Feature root directory does not exist: {feature_root}")
            check_all_datasets._printed_root_error = True
        return

    print(f"Checking feature root directory: {feature_root}")

    # Collect datasets
    datasets = []
    for item in os.listdir(feature_root):
        item_path = os.path.join(feature_root, item)
        if os.path.isdir(item_path):
            datasets.append(item)

    if not datasets:
        print("No datasets found")
        return

    print(f"Found datasets: {', '.join(datasets)}")

    all_results = {}
    total_complete = 0
    total_incomplete = 0

    for dataset_name in datasets:
        complete, incomplete = check_dataset_features(feature_root, dataset_name)
        all_results[dataset_name] = {
            "complete": complete,
            "incomplete": incomplete
        }
        total_complete += len(complete)
        total_incomplete += len(incomplete)

    # Overall summary
    print(f"\n{'=' * 60}")
    print("Overall summary:")
    print(f"  Complete features: {total_complete} proteins")
    print(f"  Missing features: {total_incomplete} proteins")
    print(f"  Total: {total_complete + total_incomplete} proteins")
    print(f"{'=' * 60}")

    return all_results


def delete_incomplete_proteins(feature_root, dataset_name, incomplete_proteins):
    """Delete protein folders that do not contain a full set of feature files."""
    if not incomplete_proteins:
        print(f"Dataset {dataset_name} has no incomplete proteins")
        return

    print(f"\nDeleting incomplete protein folders in dataset {dataset_name}...")

    deleted_count = 0
    for protein_info in incomplete_proteins:
        protein_id = protein_info[0]
        protein_dir = os.path.join(feature_root, dataset_name, protein_id)

        try:
            if os.path.exists(protein_dir):
                shutil.rmtree(protein_dir)
                print(f"  Deleted: {protein_id}")
                deleted_count += 1
        except Exception as e:
            print(f"  Failed to delete {protein_id}: {e}")

    print(f"Successfully deleted {deleted_count} incomplete protein folders")


def main():
    parser = argparse.ArgumentParser(description="Check feature file completeness")
    parser.add_argument(
        "--feature-root",
        type=str,
        default="Dataset/feature",
        help="Feature root directory (default: Dataset/feature)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["Train_573", "Test_129", "Test_181"],
        help="Dataset to check (default: check all)"
    )
    parser.add_argument(
        "--fix",
        type=bool,
        default=False,
        help="Delete incomplete protein folders"
    )

    args = parser.parse_args()

    # Check feature root
    if not os.path.exists(args.feature_root):
        print(f"Error: feature root directory does not exist: {args.feature_root}")
        return 1

    # Check single dataset or all datasets
    if args.dataset:
        # Single dataset
        complete, incomplete = check_dataset_features(args.feature_root, args.dataset)

        if args.fix and incomplete:
            # Delete incomplete proteins
            delete_incomplete_proteins(args.feature_root, args.dataset, incomplete)
    else:
        # All datasets
        all_results = check_all_datasets(args.feature_root)

        if args.fix and all_results:
            # Delete incomplete protein folders
            for dataset_name, results in all_results.items():
                if results["incomplete"]:
                    print(f"\nProcessing dataset: {dataset_name}")
                    delete_incomplete_proteins(
                        args.feature_root,
                        dataset_name,
                        results["incomplete"]
                    )

    return 0


if __name__ == "__main__":
    exit(main())
