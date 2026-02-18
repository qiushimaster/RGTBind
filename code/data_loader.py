import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from config import *
from utils import *

class ProteinDataset(Dataset):
    """Protein dataset based on pre-extracted feature files."""
    
    def __init__(self, dataset_name, msa_type="both", feature_base_path=None):
        """Initialize the dataset."""
        self.dataset_name = dataset_name
        self.msa_type = msa_type
        
        # Set feature root path
        if feature_base_path is None:
            self.feature_base_path = PATHS['feature_root']
        else:
            self.feature_base_path = feature_base_path
            
        self.dataset_feature_path = os.path.join(self.feature_base_path, dataset_name)
        
        # Get FASTA path and parse it
        self.fasta_file = self._get_fasta_file_path(dataset_name)
        self.binding_data = parse_binding_labels(self.fasta_file)
        
        # Get available protein list from feature folders
        self.available_proteins = self._get_available_proteins()
        
        # Validate and check feature completeness
        self.valid_proteins = self._validate_proteins()
        
        # Print dataset info only once
        if not hasattr(self, '_printed_info'):
            print(f"Dataset {dataset_name}: {len(self.valid_proteins)}/{len(self.binding_data)} proteins available with complete features")
            self._printed_info = True
    
    def _get_fasta_file_path(self, dataset_name):
        """Get the FASTA file path."""
        fasta_paths = {
            "Train_573": PATHS['train_fa'],
            "Test_129": PATHS['test_129_fa'], 
            "Test_181": PATHS['test_181_fa']
        }
        
        if dataset_name not in fasta_paths:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        fasta_file = fasta_paths[dataset_name]
        check_file_exists(fasta_file, f"{dataset_name} FASTA")
        return fasta_file
    
    def _get_available_proteins(self):
        """Get available proteins from the feature directory."""
        if not os.path.exists(self.dataset_feature_path):
            # Print only the first time
            if not hasattr(ProteinDataset._get_available_proteins, '_printed_path_warning'):
                print(f"Warning: Feature path {self.dataset_feature_path} does not exist")
                ProteinDataset._get_available_proteins._printed_path_warning = True
            return []
        
        available_proteins = []
        for protein_dir in os.listdir(self.dataset_feature_path):
            protein_path = os.path.join(self.dataset_feature_path, protein_dir)
            if os.path.isdir(protein_path):
                available_proteins.append(protein_dir)
        
        return available_proteins
    
    def _validate_proteins(self):
        """Validate proteins: present in FASTA and have complete features."""
        valid_proteins = []
        
        for protein_id in self.available_proteins:
            # Check whether the protein exists in FASTA
            if protein_id not in self.binding_data:
                # Print only the first time
                if not hasattr(ProteinDataset._validate_proteins, '_printed_fasta_warning'):
                    print(f"Warning: {protein_id} not found in FASTA file")
                    ProteinDataset._validate_proteins._printed_fasta_warning = True
                continue
            
            # Check required feature files
            protein_dir = os.path.join(self.dataset_feature_path, protein_id)
            if self._check_required_features(protein_dir, protein_id):
                # Validate feature length vs. label length
                if self._validate_feature_dimensions(protein_dir, protein_id):
                    valid_proteins.append(protein_id)
                else:
                    # Print only the first time
                    if not hasattr(ProteinDataset._validate_proteins, '_printed_dimension_warning'):
                        print(f"Warning: Feature-label dimension mismatch for {protein_id}")
                        ProteinDataset._validate_proteins._printed_dimension_warning = True
            else:
                # Print only the first time
                if not hasattr(ProteinDataset._validate_proteins, '_printed_features_warning'):
                    print(f"Warning: Missing required features for {protein_id}")
                    ProteinDataset._validate_proteins._printed_features_warning = True
        
        return valid_proteins
    
    def _check_required_features(self, protein_dir, protein_id):
        """Check whether required feature files exist."""
        # Core structural features (mandatory)
        required_features = [
            f"{protein_id}_dssp.npy",
            f"{protein_id}_dismap.npy"
        ]
        
        # Add feature requirements based on MSA type
        if self.msa_type in ['both', 'single']:
            required_features.append(f"{protein_id}_single_norm.npy")
        
        if self.msa_type in ['both', 'evo']:
            required_features.extend([
                f"{protein_id}_pssm.npy", 
                f"{protein_id}_hhm.npy"
            ])
        
        # Verify required files exist
        for feature_file in required_features:
            if not os.path.exists(os.path.join(protein_dir, feature_file)):
                return False
        
        return True
    
    def _validate_feature_dimensions(self, protein_dir, protein_id):
        """Validate that feature dimensions match label length."""
        try:
            # Load DSSP to check length
            dssp_file = os.path.join(protein_dir, f"{protein_id}_dssp.npy")
            dssp_features = np.load(dssp_file)
            
            # Get label length
            labels = self.binding_data[protein_id]['labels']
            
            # Check length consistency
            if len(dssp_features) != len(labels):
                print(f"Length mismatch for {protein_id}: DSSP={len(dssp_features)}, Labels={len(labels)}")
                return False
            
            # Check other feature files length
            if self.msa_type in ['both', 'single']:
                single_file = os.path.join(protein_dir, f"{protein_id}_single_norm.npy")
                single_features = np.load(single_file)
                if len(single_features) != len(labels):
                    return False
            
            if self.msa_type in ['both', 'evo']:
                pssm_file = os.path.join(protein_dir, f"{protein_id}_pssm.npy")
                hhm_file = os.path.join(protein_dir, f"{protein_id}_hhm.npy")
                pssm_features = np.load(pssm_file)
                hhm_features = np.load(hhm_file)
                if len(pssm_features) != len(labels) or len(hhm_features) != len(labels):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error validating dimensions for {protein_id}: {e}")
            return False
    
    def __len__(self):
        return len(self.valid_proteins)
    
    def __getitem__(self, idx):
        protein_id = self.valid_proteins[idx]
        protein_dir = os.path.join(self.dataset_feature_path, protein_id)
        
        # Load features
        node_features, distance_map = self._load_features(protein_dir, protein_id)
        
        # Get labels and sequence
        labels = self.binding_data[protein_id]['labels']
        sequence = self.binding_data[protein_id]['sequence']
        
        return {
            'protein_id': protein_id,
            'node_features': torch.from_numpy(node_features).float(),
            'distance_map': torch.from_numpy(distance_map).float(),
            'labels': torch.from_numpy(labels).float(),
            'seq_len': len(labels),
            'sequence': sequence
        }
    
    def _load_features(self, protein_dir, protein_id):
        """Load features and concatenate them in the specified order."""
        try:
            feature_list = []
            
            # 1) Load AF2 single representation if needed
            if self.msa_type in ['both', 'single']:
                single_file = os.path.join(protein_dir, f"{protein_id}_single_norm.npy")
                single_features = np.load(single_file)
                feature_list.append(single_features)
            
            # 2) Load evolutionary features if needed
            if self.msa_type in ['both', 'evo']:
                pssm_file = os.path.join(protein_dir, f"{protein_id}_pssm.npy")
                hhm_file = os.path.join(protein_dir, f"{protein_id}_hhm.npy")
                pssm_features = np.load(pssm_file)
                hhm_features = np.load(hhm_file)
                feature_list.extend([pssm_features, hhm_features])
            
            # 3) Load DSSP structural features (mandatory)
            dssp_file = os.path.join(protein_dir, f"{protein_id}_dssp.npy")
            dssp_features = np.load(dssp_file)
            feature_list.append(dssp_features)
            
            # 4) Load distance map
            dismap_file = os.path.join(protein_dir, f"{protein_id}_dismap.npy")
            distance_map = np.load(dismap_file)
            
            # Concatenate node features in order: single, pssm, hhm, dssp
            node_features = np.hstack(feature_list)
            
            return node_features, distance_map
            
        except Exception as e:
            print(f"Error loading features for {protein_id}: {e}")
            raise e
    
    def get_feature_dim(self):
        """Get the total feature dimension."""
        total_dim = 0
        
        if self.msa_type in ['both', 'single']:
            total_dim += FEATURE_DIMS['single']  # 384
        
        if self.msa_type in ['both', 'evo']:
            total_dim += FEATURE_DIMS['pssm']    # 20
            total_dim += FEATURE_DIMS['hhm']     # 20
        
        total_dim += FEATURE_DIMS['dssp']        # 14
        
        return total_dim

def collate_fn(batch):
    """Batch collation function (kept consistent with the author's implementation)."""
    batch_size = len(batch)
    max_len = max([item['seq_len'] for item in batch])
    feature_dim = batch[0]['node_features'].shape[1]
    
    # Create batch tensors with padding
    batch_node_features = torch.zeros(batch_size, max_len, feature_dim)
    batch_distance_maps = torch.zeros(batch_size, max_len, max_len)
    batch_labels = torch.zeros(batch_size, max_len)
    batch_masks = torch.zeros(batch_size, max_len)
    batch_protein_ids = []
    batch_seq_lens = []
    batch_sequences = []
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        batch_node_features[i, :seq_len] = item['node_features']
        batch_distance_maps[i, :seq_len, :seq_len] = item['distance_map']
        batch_labels[i, :seq_len] = item['labels']
        batch_masks[i, :seq_len] = 1  # valid positions
        batch_protein_ids.append(item['protein_id'])
        batch_seq_lens.append(seq_len)
        batch_sequences.append(item['sequence'])
    
    return {
        'node_features': batch_node_features,
        'distance_maps': batch_distance_maps,
        'labels': batch_labels,
        'masks': batch_masks,
        'protein_ids': batch_protein_ids,
        'seq_lens': batch_seq_lens,
        'sequences': batch_sequences
    }

class DataManager:
    """Data manager aligned with the updated feature-loading logic."""
    
    def __init__(self, msa_type="both", feature_base_path=None):
        self.msa_type = msa_type
        self.feature_base_path = feature_base_path
        self.datasets = {}
        self.k_fold_splits = None
        
    def load_dataset(self, dataset_name):
        """Load a dataset."""
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = ProteinDataset(
                dataset_name=dataset_name,
                msa_type=self.msa_type,
                feature_base_path=self.feature_base_path
            )
        return self.datasets[dataset_name]
    
    def create_k_fold_splits(self, dataset_name="Train_573", k=5):
        """Create K-fold splits."""
        dataset = self.load_dataset(dataset_name)
        protein_ids = dataset.valid_proteins
        self.k_fold_splits = split_dataset_k_fold(protein_ids, k, TRAIN_CONFIG['seed'])
        
        # Print split information
        for i, fold in enumerate(self.k_fold_splits):
            print(f"Fold {i}: Train={len(fold['train'])}, Test={len(fold['test'])}")
        
        return self.k_fold_splits
    
    def get_dataloader(self, dataset_name, protein_ids=None, batch_size=16, shuffle=True):
        """Get a DataLoader."""
        dataset = self.load_dataset(dataset_name)
        
        if protein_ids is not None:
            # Create a subset dataset
            filtered_dataset = self._create_filtered_dataset(dataset, protein_ids)
        else:
            filtered_dataset = dataset
        
        dataloader = DataLoader(
            dataset=filtered_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=DATASET_CONFIG['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return dataloader
    
    def _create_filtered_dataset(self, original_dataset, protein_ids):
        """Create a filtered (subset) dataset."""
        class FilteredDataset(Dataset):
            def __init__(self, original_dataset, protein_ids):
                self.original_dataset = original_dataset
                self.protein_ids = protein_ids
                # Find indices of valid proteins in the original dataset
                self.valid_indices = []
                for protein_id in protein_ids:
                    if protein_id in original_dataset.valid_proteins:
                        idx = original_dataset.valid_proteins.index(protein_id)
                        self.valid_indices.append(idx)
                
                print(f"Filtered dataset: {len(self.valid_indices)}/{len(protein_ids)} proteins found")
            
            def __len__(self):
                return len(self.valid_indices)
            
            def __getitem__(self, idx):
                original_idx = self.valid_indices[idx]
                return self.original_dataset[original_idx]
        
        return FilteredDataset(original_dataset, protein_ids)
    
    def get_cv_dataloaders(self, fold_idx, batch_size=16):
        """Get cross-validation DataLoaders."""
        if self.k_fold_splits is None:
            self.create_k_fold_splits()
        
        train_ids = self.k_fold_splits[fold_idx]['train']
        val_ids = self.k_fold_splits[fold_idx]['test']
        
        train_loader = self.get_dataloader("Train_573", train_ids, batch_size, shuffle=True)
        val_loader = self.get_dataloader("Train_573", val_ids, batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def get_test_dataloader(self, dataset_name, batch_size=16):
        """Get a test DataLoader."""
        return self.get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
    

def create_sampled_dataloader(dataset, samples_per_epoch=5000, batch_size=16, seed=None):
    """Create a sampled DataLoader."""
    
    # Create RandomSampler consistent with the author's implementation
    sampler = RandomSampler(dataset, replacement=True, num_samples=samples_per_epoch)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # when using sampler, shuffle must be False
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=DATASET_CONFIG['num_workers']
    )
