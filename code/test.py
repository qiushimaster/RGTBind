import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from model import create_rgt_model
from config import *
from utils import *
from data_loader import DataManager, create_sampled_dataloader
import argparse
import itertools
from tqdm import tqdm
import copy

class GraphSiteTrainer:
    """Trainer"""

    def __init__(self, msa_type="both", device=None, config_override=None, output_dir=None, setup_logging=True):
        self.msa_type = msa_type
        self.device = device if device else get_device()
        self.data_manager = DataManager(msa_type)

        # Deep-copy configuration overrides
        self.config_override = copy.deepcopy(config_override) if config_override else {}

        # Create timestamped output directory
        if output_dir is None:
            self.output_dir, self.timestamp = create_timestamped_dir(
                PATHS['results_root'], msa_type, 'train')
        else:
            self.output_dir = output_dir
            self.timestamp = get_timestamp()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Determine feature dimension
        self.node_dim = self._calculate_node_dim()

        # Store CV predictions for threshold selection
        self.cv_predictions_list = []
        self.cv_labels_list = []

        # CSV file path
        self.results_csv = os.path.join(self.output_dir, "results.csv")

        # Setup logging
        if setup_logging:
            self.setup_logging()
        else:
            self.logger = None

    def _calculate_node_dim(self):
        """Compute node feature dimension."""
        return calculate_node_dim(self.msa_type)

    def get_effective_config(self):
        """Get effective configuration."""
        effective_config = copy.deepcopy(TRAIN_CONFIG)
        effective_config.update(copy.deepcopy(self.config_override))
        return effective_config

    def _is_using_fixed_threshold(self):
        """Whether fixed-threshold mode is enabled."""
        effective_config = self.get_effective_config()
        return effective_config.get('use_fixed_threshold', False)

    def _get_threshold_value(self):
        """Get threshold value (fixed or default)."""
        effective_config = self.get_effective_config()
        if self._is_using_fixed_threshold():
            return effective_config.get('default_threshold', get_default_threshold(self.msa_type))
        return None

    def _calculate_metrics_with_config(self, labels, predictions, threshold_override=None):
        """Compute metrics according to configuration."""
        use_fixed = self._is_using_fixed_threshold()
        threshold = threshold_override if threshold_override is not None else self._get_threshold_value()

        return calculate_metrics(
            labels, None, predictions,
            threshold=threshold,
            use_fixed_threshold=use_fixed,
            msa_type=self.msa_type
        )

    def setup_logging(self):
        """Setup logging."""
        # Log file
        log_file = os.path.join(self.output_dir, f'training_{self.timestamp}.log')
        # Create logger
        logger_name = f"graphsite_trainer_{id(self)}"
        self.logger = setup_logger(logger_name, log_file)

        # Log initialization info
        self._log_initialization_info()

    def _log_initialization_info(self):
        """Log initialization info."""
        if not hasattr(self, '_logged_init'):
            self.logger.info(f"Training session started at {self.timestamp}")
            self.logger.info(f"MSA type: {self.msa_type}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Output directory: {self.output_dir}")

            effective_config = self.get_effective_config()
            self.logger.info(f"Model configuration: {MODEL_CONFIG}")
            self.logger.info(f"Training configuration: {effective_config}")

            if self.config_override:
                self.logger.info(f"Configuration overrides: {self.config_override}")

            # Log threshold mode
            if self._is_using_fixed_threshold():
                threshold = self._get_threshold_value()
                self.logger.info(f"Using fixed threshold mode: {threshold:.4f}")
            else:
                self.logger.info("Using dynamic threshold mode (determined by F1-score maximization)")

            self._logged_init = True

    def log_info(self, message):
        """Write log."""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"INFO: {message}")

    def calibrate_model_temperature(self, model, val_loader):
        return

    def create_model(self):
        """Create model."""
        effective_config = self.get_effective_config()
        # Merge model and training config
        model_cfg = {**MODEL_CONFIG, **effective_config}

        # Log the config passed into the model
        self.log_info(f"Model configuration (effective): {model_cfg}")

        model = create_rgt_model(self.device, model_cfg)
        self.log_info(f"Model created with {count_parameters(model):,} parameters")

        return model

    def create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        effective_config = self.get_effective_config()
        scheduler_type = effective_config.get('lr_scheduler', 'cosine')

        if scheduler_type == 'step':
            step_size = effective_config.get('step_size', 3)
            gamma = effective_config.get('step_gamma', 0.5)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            self.log_info(f"Using StepLR scheduler with step_size={step_size}, gamma={gamma}")

        elif scheduler_type == 'cosine':
            max_epochs = effective_config.get('max_epochs', 15)
            min_lr = effective_config.get('min_lr', 1e-6)
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)
            self.log_info(f"Using CosineAnnealingLR scheduler with T_max={max_epochs}, eta_min={min_lr}")

        elif scheduler_type == 'custom_step':
            # Custom stepwise learning-rate schedule
            epochs_list = effective_config.get('custom_step_epochs', [])
            lrs_list = effective_config.get('custom_step_lrs', [])
            if len(epochs_list) != len(lrs_list):
                raise ValueError("custom_step_epochs and custom_step_lrs must have the same length")
            # Log
            self.log_info(f"Using CustomStep scheduler with epochs={epochs_list}, lrs={lrs_list}")
            scheduler = None

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _apply_custom_step_lr(self, optimizer, epoch_index_zero_based):
        """Apply the configured LR at the specified epoch (custom_step)."""
        effective_config = self.get_effective_config()
        base_lr = effective_config['learning_rate']
        epochs_list = effective_config.get('custom_step_epochs', [])
        lrs_list = effective_config.get('custom_step_lrs', [])
        # Current epoch index is 0-based; convert to 1-based
        current_epoch_1based = epoch_index_zero_based + 1
        # Compute target LR
        target_lr = base_lr
        for step_epoch, step_lr in zip(epochs_list, lrs_list):
            if current_epoch_1based >= step_epoch:
                target_lr = step_lr
        for group in optimizer.param_groups:
            group['lr'] = target_lr
        return target_lr

    def _extract_valid_predictions_and_labels(self, outputs, labels, masks):
        """Extract valid predictions and labels."""
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs

        sigmoid_outputs = torch.sigmoid(logits)
        valid_mask = masks.bool()
        valid_predictions = sigmoid_outputs[valid_mask]
        valid_labels = labels[valid_mask]

        predictions = valid_predictions.detach().cpu().numpy()
        labels_np = valid_labels.detach().cpu().numpy()

        return predictions, labels_np

    def train_one_epoch(self, model, dataloader, optimizer, scheduler, epoch):
        """Train one epoch."""
        model.train()
        total_loss = 0
        total_main_loss = 0
        n_batches = 0
        all_predictions = []
        all_labels = []

        start_time = time.time()

        # Use progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            node_features = batch['node_features'].to(self.device)
            distance_maps = batch['distance_maps'].to(self.device)
            labels = batch['labels'].to(self.device)
            masks = batch['masks'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(node_features, None, distance_maps, masks)

            # Compute loss using model.compute_loss
            loss_dict = model.compute_loss(outputs, labels, masks)
            total_loss_value = loss_dict['total_loss']

            # Backpropagation
            total_loss_value.backward()
            optimizer.step()

            # Track losses
            total_loss += total_loss_value.item()
            total_main_loss += loss_dict['main_loss'].item()
            n_batches += 1

            # Collect predictions
            with torch.no_grad():
                predictions, labels_np = self._extract_valid_predictions_and_labels(outputs, labels, masks)
                all_predictions.append(predictions)
                all_labels.append(labels_np)

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{total_loss_value.item():.4f}',
                'Avg_Loss': f'{total_loss/n_batches:.4f}',
                'LR': f'{current_lr:.2e}'
            })

        # Step scheduler
        if scheduler:
            scheduler.step()

        avg_loss = total_loss / n_batches
        avg_main_loss = total_main_loss / n_batches
        training_time = time.time() - start_time

        # Compute overall metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        # Use unified metrics
        metrics = self._calculate_metrics_with_config(all_labels, all_predictions)

        # Log training info
        final_lr = optimizer.param_groups[0]['lr']
        log_msg = (f"Epoch {epoch} Training - Loss: {avg_loss:.4f}, "
                  f"AUPR: {metrics['AUPR']:.4f}, AUC: {metrics['AUC']:.4f}, "
                  f"MCC: {metrics['MCC']:.4f}, LR: {final_lr:.2e}, Time: {format_time(training_time)}")
        self.log_info(log_msg)

        return avg_loss, metrics

    def validate(self, model, dataloader):
        """Validate."""
        model.eval()

        total_loss = 0
        total_main_loss = 0
        n_batches = 0
        all_predictions = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation", leave=False)

            for batch in pbar:
                node_features = batch['node_features'].to(self.device)
                distance_maps = batch['distance_maps'].to(self.device)
                labels = batch['labels'].to(self.device)
                masks = batch['masks'].to(self.device)

                outputs = model(node_features, None, distance_maps, masks)

                # Compute loss using model.compute_loss
                loss_dict = model.compute_loss(outputs, labels, masks)

                total_loss += loss_dict['total_loss'].item()
                total_main_loss += loss_dict['main_loss'].item()
                n_batches += 1

                # Collect predictions
                predictions, labels_np = self._extract_valid_predictions_and_labels(outputs, labels, masks)
                all_predictions.append(predictions)
                all_labels.append(labels_np)

                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss_dict["total_loss"].item():.4f}'})

        avg_loss = total_loss / n_batches
        avg_main_loss = total_main_loss / n_batches
        validation_time = time.time() - start_time

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        metrics = self._calculate_metrics_with_config(all_labels, all_predictions)

        # Log validation info
        log_msg = (f"Validation - Loss: {avg_loss:.4f}, "
                  f"AUPR: {metrics['AUPR']:.4f}, AUC: {metrics['AUC']:.4f}, "
                  f"MCC: {metrics['MCC']:.4f}, Time: {format_time(validation_time)}")
        self.log_info(log_msg)

        return avg_loss, metrics, all_predictions, all_labels

    def train_fold(self, fold_idx, save_model=True):
        """Train one fold."""

        self.log_info(f"{'='*60}")
        self.log_info(f"Training Fold {fold_idx + 1}/{TRAIN_CONFIG['num_folds']}")
        self.log_info(f"{'='*60}")

        effective_config = self.get_effective_config()

        # Get dataloaders
        train_loader, val_loader = self.data_manager.get_cv_dataloaders(
            fold_idx, effective_config['batch_size'])

        # Create sampled training dataloader
        train_dataset = train_loader.dataset
        sampled_train_loader = create_sampled_dataloader(
            train_dataset,
            effective_config['samples_per_epoch'],
            effective_config['batch_size'],
            effective_config['seed'] + fold_idx
        )

        # Create model
        model = self.create_model()

        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=effective_config['learning_rate'],
            betas=(effective_config['beta1'], effective_config['beta2']),
            eps=effective_config['eps'],
            weight_decay=effective_config['weight_decay']
        )

        # Create LR scheduler
        scheduler = self.create_scheduler(optimizer)

        # Early stopping - based on AUPR
        early_stopping = EarlyStopping(
            patience=effective_config['patience'],
            restore_best_weights=True
        )

        start_time = time.time()

        # Epoch progress bar
        epoch_pbar = tqdm(range(effective_config['max_epochs']), desc=f"Fold {fold_idx + 1}")

        # Track best history
        best_val_metrics = None
        best_epoch = 0
        best_model_state = None

        for epoch in epoch_pbar:
            self.log_info(f"Epoch {epoch + 1}/{effective_config['max_epochs']}")

            # For custom_step, set LR before epoch starts
            if effective_config.get('lr_scheduler') == 'custom_step':
                new_lr = self._apply_custom_step_lr(optimizer, epoch)
                self.log_info(f"CustomStep LR applied for epoch {epoch + 1}: {new_lr:.2e}")

            # Train
            train_loss, train_metrics = self.train_one_epoch(
                model, sampled_train_loader, optimizer, scheduler, epoch + 1)

            # Validate
            val_loss, val_metrics, val_predictions, val_labels = self.validate(model, val_loader)

            # Save the model/metrics for best AUPR; also (dynamic threshold mode) store current optimal threshold
            if best_val_metrics is None or val_metrics['AUPR'] > best_val_metrics['AUPR']:
                best_val_metrics = val_metrics.copy()
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()

                # If using dynamic threshold, compute and temporarily store best threshold for this fold
                if not self._is_using_fixed_threshold():
                    try:
                        fold_opt_threshold = determine_cv_threshold([val_predictions], [val_labels])
                        self._last_fold_opt_threshold = fold_opt_threshold
                    except Exception as e:
                        self.log_info(f"Warning: failed to determine fold threshold at epoch {best_epoch}: {e}")

                # Save best model
                if save_model:
                    model_save_path = os.path.join(self.output_dir, f"fold{fold_idx}.ckpt")
                    torch.save(best_model_state, model_save_path)
                    self.log_info(f"New best model saved for fold {fold_idx} at epoch {best_epoch} (AUPR: {best_val_metrics['AUPR']:.4f})")

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_loss:.4f}',
                'Val_Loss': f'{val_loss:.4f}',
                'Val_AUPR': f'{val_metrics["AUPR"]:.4f}',
                'Best_AUPR': f'{best_val_metrics["AUPR"]:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            self.log_info("-" * 40)

            # Early-stopping check - based on AUPR
            if early_stopping(val_metrics['AUPR'], model):
                self.log_info(f"Early stopping at epoch {epoch + 1}")
                break

        training_time = time.time() - start_time
        if not hasattr(self, '_printed_training_completion'):
            self.log_info(f"Training completed in {format_time(training_time)}")
            self._printed_training_completion = True

        # Final validation eval - using best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        final_val_metrics = best_val_metrics

        # Collect predictions for threshold selection (dynamic threshold mode only)
        if not self._is_using_fixed_threshold():
            # Re-run validation to obtain predictions/labels for threshold determination
            _, _, final_val_predictions, final_val_labels = self.validate(model, val_loader)
            self.cv_predictions_list.append(final_val_predictions)
            self.cv_labels_list.append(final_val_labels)

        # Save fold result to CSV using best metrics
        fold_result = {
            'fold': fold_idx,
            'msa_type': self.msa_type,
            'training_time': training_time,
            'best_epoch': best_epoch,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'timestamp': self.timestamp,
            'model_path': self.output_dir,
            'use_fixed_threshold': self._is_using_fixed_threshold()
        }
        fold_result.update({f'val_{k}': v for k, v in final_val_metrics.items()})  # best metrics
        fold_result.update(effective_config)

        save_results_to_csv(fold_result, self.results_csv, mode='a')

        # Return best metrics and the corresponding best threshold for this fold (dynamic threshold)
        best_threshold_for_fold = getattr(self, '_last_fold_opt_threshold', None) if not self._is_using_fixed_threshold() else self._get_threshold_value()
        # Clean up temporary attribute to avoid leaking across folds
        if hasattr(self, '_last_fold_opt_threshold'):
            delattr(self, '_last_fold_opt_threshold')
        return model, final_val_metrics, best_threshold_for_fold

    def cross_validation_training(self):
        """Run cross-validation training."""
        effective_config = self.get_effective_config()

        # Basic info
        if not hasattr(self, '_printed_cv_start'):
            self.log_info(f"Starting {TRAIN_CONFIG['num_folds']}-fold cross-validation training")
            self.log_info(f"MSA type: {self.msa_type}")
            self.log_info(f"Device: {self.device}")
            self._printed_cv_start = True

        # Log threshold mode
        if self._is_using_fixed_threshold():
            threshold = self._get_threshold_value()
            self.log_info(f"Using fixed threshold: {threshold:.4f}")
        else:
            self.log_info("Using dynamic threshold (will be determined from CV results)")

        # Load training data
        dataset = self.data_manager.load_dataset("Train_573")
        # Print dataset info only once
        if not hasattr(self, '_printed_dataset_info'):
            self.log_info(f"Training dataset loaded: {len(dataset.valid_proteins)} proteins")
            self._printed_dataset_info = True

        # Create K-fold splits
        self.data_manager.create_k_fold_splits()

        # Train folds
        fold_metrics = []

        total_start_time = time.time()

        # Fold progress bar
        fold_pbar = tqdm(range(TRAIN_CONFIG['num_folds']), desc="Cross-Validation")

        fold_best_thresholds = []
        for fold_idx in fold_pbar:
            model, metrics, fold_threshold = self.train_fold(fold_idx)
            fold_metrics.append(metrics)
            fold_best_thresholds.append(fold_threshold)

            # Update fold progress bar
            fold_pbar.set_postfix({
                'AUPR': f'{metrics["AUPR"]:.4f}',
                'AUC': f'{metrics["AUC"]:.4f}',
                'MCC': f'{metrics["MCC"]:.4f}'
            })

            # Detailed fold info
            self.log_info(f"\nFold {fold_idx + 1} Results:")
            self.log_info(f"AUPR: {metrics['AUPR']:.4f}, AUC: {metrics['AUC']:.4f}, "
                           f"MCC: {metrics['MCC']:.4f}, F1: {metrics['F1']:.4f}")

        total_training_time = time.time() - total_start_time
        if not hasattr(self, '_printed_total_time'):
            self.log_info(f"\nTotal training time: {format_time(total_training_time)}")
            self._printed_total_time = True

        avg_metrics = {}
        std_metrics = {}
        for metric in METRICS:
            if metric in fold_metrics[0]:
                values = [fm[metric] for fm in fold_metrics]
                avg_metrics[metric] = float(np.mean(values))
                std_metrics[metric] = float(np.std(values))
        cv_metrics_with_threshold = avg_metrics

        if not hasattr(self, '_printed_cv_results'):
            self.log_info("\nCross-Validation Average Results (final):")
            for metric in METRICS:
                if metric in cv_metrics_with_threshold:
                    self.log_info(f"{metric}: {cv_metrics_with_threshold[metric]:.4f} ± {std_metrics[metric]:.4f}")
            self._printed_cv_results = True

        # Threshold: fixed -> from config; dynamic -> best-AUPR fold threshold
        if self._is_using_fixed_threshold():
            optimal_threshold = self._get_threshold_value()
        else:
            best_fold_idx = int(np.argmax([m['AUPR'] for m in fold_metrics]))
            optimal_threshold = fold_best_thresholds[best_fold_idx]
            threshold_csv = os.path.join(self.output_dir, "optimal_threshold.csv")
            save_threshold_to_csv(optimal_threshold, threshold_csv, self.msa_type)
            self.log_info(f"Optimal threshold saved to: {threshold_csv}")

        # Save CV summary to CSV
        cv_summary = {
            'result_type': 'cv_summary',
            'msa_type': self.msa_type,
            'num_folds': TRAIN_CONFIG['num_folds'],
            'optimal_threshold': optimal_threshold,
            'use_fixed_threshold': self._is_using_fixed_threshold(),
            'training_time': total_training_time,
            'timestamp': self.timestamp,
            'model_path': self.output_dir
        }
        cv_summary.update({f'cv_{k}': v for k, v in cv_metrics_with_threshold.items()})
        cv_summary.update({f'std_{k}': v for k, v in std_metrics.items()})
        cv_summary.update(effective_config)

        save_results_to_csv(cv_summary, self.results_csv, mode='a')

        return cv_metrics_with_threshold

    def grid_search(self, lr_scheduler='cosine', param_grid=None):
        """Run grid search."""
        # Get parameter grid based on scheduler type
        if param_grid is None:
            param_grid = get_grid_search_config(lr_scheduler)

        # Print grid search info
        if not hasattr(self, '_printed_gs_start'):
            self.log_info("Starting grid search for hyperparameter optimization")
            self.log_info(f"Learning rate scheduler: {lr_scheduler}")
            self.log_info(f"Parameter grid: {param_grid}")

            # Generate all parameter combinations
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(itertools.product(*param_values))

            self.log_info(f"Total combinations to test: {len(param_combinations)}")
            self._printed_gs_start = True
        else:
            # Generate all parameter combinations
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(itertools.product(*param_values))

        best_score = -np.inf
        best_params = None

        # Grid search results CSV
        gs_csv_file = os.path.join(self.output_dir, f'grid_search_results_{lr_scheduler}.csv')

        for i, param_combination in enumerate(param_combinations):
            if not hasattr(self, '_printed_separator'):
                self.log_info(f"\n{'='*80}")
                self.log_info(f"Grid Search {i+1}/{len(param_combinations)}")
                self.log_info(f"{'='*80}")
                self._printed_separator = True

            # Create parameter dict
            current_params = dict(zip(param_names, param_combination))
            # Add scheduler type
            current_params['lr_scheduler'] = lr_scheduler

            # Ensure threshold-related config is correctly passed through
            if 'use_fixed_threshold' in self.config_override:
                current_params['use_fixed_threshold'] = self.config_override['use_fixed_threshold']
            if 'default_threshold' in self.config_override:
                current_params['default_threshold'] = self.config_override['default_threshold']

            if not hasattr(self, '_printed_params'):
                self.log_info(f"Testing parameters: {current_params}")
                self._printed_params = True

            try:
                # Create directory name
                param_dir_name = self._create_param_dir_name(lr_scheduler, current_params)
                param_dir = os.path.join(self.output_dir, f"gs_{param_dir_name}")

                # Create a new trainer instance
                trainer = GraphSiteTrainer(
                    msa_type=self.msa_type,
                    device=self.device,
                    config_override=current_params,
                    output_dir=param_dir,
                    setup_logging=True
                )

                # Run cross-validation
                cv_metrics = trainer.cross_validation_training()

                # Record result
                score = cv_metrics['AUPR']  # Use AUPR as the evaluation metric

                # Save to grid-search CSV
                gs_result = self._create_grid_search_result(
                    i+1, lr_scheduler, current_params, cv_metrics, score, 'success', ''
                )
                self._save_grid_search_result(gs_result, gs_csv_file)

                if not hasattr(self, '_printed_gs_combination'):
                    self.log_info(f"Combination {i+1} - AUPR: {score:.4f}")
                    self._printed_gs_combination = True

                # Update best result
                if score > best_score:
                    best_score = score
                    best_params = current_params.copy()
                    # Only print when best updates
                    self.log_info(f"New best score: {best_score:.4f}")

            except Exception as e:
                self.log_info(f"Error in combination {i+1}: {str(e)}")

                # Save error to grid-search CSV
                gs_result = self._create_grid_search_result(
                    i+1, lr_scheduler, current_params, {}, -1.0, 'error', str(e)
                )
                self._save_grid_search_result(gs_result, gs_csv_file)

        self._log_grid_search_completion(lr_scheduler, best_params, best_score, gs_csv_file)

        return best_params, best_score

    def _create_param_dir_name(self, lr_scheduler, current_params):
        """Create parameter directory name."""
        param_abbreviations = {
            'batch_size': 'bs',
            'learning_rate': 'lr',
            'weight_decay': 'wd',
            'dropout_rate': 'dr',
            'min_lr': 'minlr',   # cosine annealing params
            'step_gamma': 'sg',  # step scheduler params
            'step_size': 'ss'    # step scheduler params
        }
        param_dir_parts = [f"sch_{lr_scheduler}"]  # scheduler prefix
        for k, v in current_params.items():
            if k not in ['lr_scheduler', 'use_fixed_threshold', 'default_threshold']:
                abbr = param_abbreviations.get(k, k)
                param_dir_parts.append(f"{abbr}_{v}")
        return "_".join(param_dir_parts)

    def _create_grid_search_result(self, combination_id, lr_scheduler, current_params, cv_metrics, score, status, error_message):
        """Create a grid-search result dict."""
        gs_result = {
            'combination_id': combination_id,
            'scheduler': lr_scheduler,
            'score': score,
            'status': status,
            'error_message': error_message,
            'timestamp': self.timestamp
        }
        gs_result.update(current_params)

        if cv_metrics:
            gs_result.update({f'cv_{k}': v for k, v in cv_metrics.items()})
        else:
            # Fill empty CV metrics
            for metric in METRICS:
                gs_result[f'cv_{metric}'] = -1.0

        return gs_result

    def _save_grid_search_result(self, gs_result, gs_csv_file):
        """Save grid-search result to CSV."""
        try:
            save_results_to_csv(gs_result, gs_csv_file, mode='a')
        except Exception as csv_error:
            self.log_info(f"Failed to save CSV result: {str(csv_error)}")

    def _log_grid_search_completion(self, lr_scheduler, best_params, best_score, gs_csv_file):
        """Log grid-search completion info."""
        # Print completion info
        if not hasattr(self, '_printed_completion'):
            self.log_info(f"\n{'='*80}")
            self.log_info("GRID SEARCH COMPLETED")
            self.log_info(f"{'='*80}")
            self.log_info(f"Scheduler type: {lr_scheduler}")
            self.log_info(f"Best parameters: {best_params}")
            self.log_info(f"Best AUPR score: {best_score:.4f}")
            self.log_info(f"All results saved to: {gs_csv_file}")
            self._printed_completion = True


def main():
    parser = argparse.ArgumentParser(description="Train GraphSite model")
    parser.add_argument("--msa", type=str, default="both",
                       choices=["both", "single", "evo"],
                       help="MSA information type")
    parser.add_argument("--grid_search", type=bool, default=False,
                       help="Whether to run grid search for hyperparameter optimization")
    parser.add_argument("--use_fixed_threshold", type=bool, default=False,
                       help="Whether to use a fixed threshold")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       choices=["cosine", "step", "custom_step"],
                       help="Learning-rate scheduler type")

    args = parser.parse_args()
    create_directories()

    # Set random seed
    seed_everything(TRAIN_CONFIG['seed'])

    # Create config overrides
    config_override = {
        'lr_scheduler': args.lr_scheduler,
    }

    # Threshold-related parameters
    if args.use_fixed_threshold:
        config_override['use_fixed_threshold'] = True
        config_override['default_threshold'] = get_default_threshold(args.msa)
    else:
        config_override['use_fixed_threshold'] = False

    # Remove None values
    config_override = {k: v for k, v in config_override.items() if v is not None}

    # Create trainer
    setup_main_logging = not args.grid_search
    trainer = GraphSiteTrainer(
        msa_type=args.msa,
        config_override=config_override,
        setup_logging=setup_main_logging
    )

    # Log start info
    if setup_main_logging:
        trainer.log_info(f"Starting GraphSite training")
        trainer.log_info(f"MSA type: {args.msa}")
        trainer.log_info(f"Configuration: {MODEL_CONFIG}")
        trainer.log_info(f"Training config: {trainer.get_effective_config()}")
    else:
        if not hasattr(main, '_printed_gs_info'):
            print(f"Starting GraphSite grid search")
            print(f"MSA type: {args.msa}")
            print(f"Scheduler: {args.lr_scheduler}")
            main._printed_gs_info = True

    # Run selected mode
    if args.grid_search:
        # Grid search
        best_params, best_score = trainer.grid_search(lr_scheduler=args.lr_scheduler)

        # Completion info
        if not hasattr(main, '_printed_gs_completion'):
            print("\n" + "="*60)
            print("GRID SEARCH COMPLETED")
            print("="*60)
            print(f"Scheduler type: {args.lr_scheduler}")
            print(f"Best parameters: {best_params}")
            print(f"Best AUPR score: {best_score:.4f}")
            main._printed_gs_completion = True

    else:
        # Always run cross-validation training
        cv_metrics = trainer.cross_validation_training()

        # Completion info
        if not hasattr(trainer, '_printed_cv_completion'):
            trainer.log_info("\n" + "="*60)
            trainer.log_info("CROSS-VALIDATION TRAINING COMPLETED")
            trainer.log_info("="*60)
            trainer.log_info(f"Average AUC: {cv_metrics['AUC']:.4f}")
            trainer.log_info(f"Average AUPR: {cv_metrics['AUPR']:.4f}")
            trainer.log_info(f"Average MCC: {cv_metrics['MCC']:.4f}")
            trainer.log_info(f"Average F1: {cv_metrics['F1']:.4f}")

            # Threshold info
            if trainer._is_using_fixed_threshold():
                threshold = trainer._get_threshold_value()
                trainer.log_info(f"Fixed threshold used: {threshold:.4f}")
            else:
                trainer.log_info("Dynamic threshold determined from CV results")

            trainer._printed_cv_completion = True

    # Output save info
    if setup_main_logging:
        trainer.log_info(f"All outputs saved to: {trainer.output_dir}")
        trainer.log_info(f"Results CSV: {trainer.results_csv}")
    else:
        if not hasattr(main, '_printed_save_info'):
            print(f"All outputs saved to: {trainer.output_dir}")
            print(f"Results CSV: {trainer.results_csv}")
            main._printed_save_info = True


if __name__ == "__main__":
    main()
