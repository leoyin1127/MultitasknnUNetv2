#!/usr/bin/env python
"""
Enhanced MultitasknnUNetTrainer with major fixes to resolve segmentation and classification issues:
1. Improved architecture with gradient isolation
2. Optimized training phases with longer segmentation focus
3. Balanced loss functions with better weighting scheme
4. Enhanced classification head with proper regularization
"""

# Suppress TorchDynamo errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import os
import re
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List, Dict, Optional
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import (
    join, load_json, save_json, isfile, maybe_mkdir_p
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder

import SimpleITK as sitk

###############################################################################
# Optimized Simple Classification Head with Strong Regularization
###############################################################################
class SimpleClassificationHead(nn.Module):
    """
    Streamlined classification head with proper regularization and simpler architecture.
    Designed to minimize interference with segmentation task.
    """
    def __init__(self, in_channels: int, num_classes: int = 3, dropout_rate: float = 0.3):
        super(SimpleClassificationHead, self).__init__()

        # Global average pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Layer normalization for better stability
        self.norm = nn.LayerNorm(in_channels)

        # Simple but effective MLP with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        # Initialize weights carefully
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to improve early training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Apply layer normalization
        x = self.norm(x)

        # Classification
        return self.classifier(x)

###############################################################################
# Fixed MultitaskUNet with Improved Gradient Isolation
###############################################################################
class MultitaskUNet(nn.Module):
    """
    Improved UNet with better separation between segmentation and classification.
    Features cleaner gradient flow and explicit handling of task interference.
    """
    def __init__(self, base_network: nn.Module, num_classes: int = 3):
        super(MultitaskUNet, self).__init__()

        # Store base network for segmentation
        self.base_network = base_network

        # Detect bottleneck dimension
        bottleneck_dim = 320  # Default fallback

        # Try to determine bottleneck channels from network architecture
        if hasattr(base_network, 'encoder') and hasattr(base_network.encoder, 'stages'):
            encoder = base_network.encoder
            if len(encoder.stages) > 0:
                bottleneck_dim = encoder.stages[-1].output_channels
                print(f"[MultitaskUNet] Detected bottleneck dimension: {bottleneck_dim}")

        # Create a simpler classification head
        self.classification_head = SimpleClassificationHead(bottleneck_dim, num_classes)

        # Store reference to decoder and encoder for convenience
        if hasattr(base_network, 'decoder'):
            self.decoder = base_network.decoder
        if hasattr(base_network, 'encoder'):
            self.encoder = base_network.encoder

        # For storing intermediate outputs
        self.last_classification_output = None
        self.bottleneck_features = None

        # Training vs inference mode
        self.training_phase = 1  # 1: segmentation only, 2: combined
        self.inference_mode = False

        print(f"[MultitaskUNet] Initialized with improved gradient isolation")
        print(f"[MultitaskUNet] Classification head input dimensions: {bottleneck_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with improved gradient isolation.
        """
        batch_size = x.shape[0]

        # Phase 1: Focus on segmentation (minimal classification)
        if self.training and self.training_phase == 1:
            # Run segmentation network
            seg_output = self.base_network(x)

            # Run classification with very minimal backprop impact
            with torch.set_grad_enabled(False):  # No gradients for encoder during classification
                if hasattr(self.base_network, 'encoder'):
                    encoder_features = self.base_network.encoder(x)

                    if isinstance(encoder_features, list):
                        bottleneck_features = encoder_features[-1]
                    else:
                        bottleneck_features = encoder_features

                    self.bottleneck_features = bottleneck_features.detach()  # Explicit detach

                    # Run classification on detached features (no gradient to encoder)
                    self.last_classification_output = self.classification_head(self.bottleneck_features)
                else:
                    self.last_classification_output = torch.zeros(batch_size, 3, device=x.device)

            return seg_output

        # Inference mode: efficient single pass
        elif self.inference_mode or not self.training:
            seg_output = self.base_network(x)

            if hasattr(self.base_network, 'encoder'):
                # Reuse encoder features if possible
                encoder_features = self.base_network.encoder(x)

                if isinstance(encoder_features, list):
                    bottleneck_features = encoder_features[-1]
                else:
                    bottleneck_features = encoder_features

                self.bottleneck_features = bottleneck_features
                self.last_classification_output = self.classification_head(bottleneck_features)
            else:
                self.last_classification_output = torch.zeros(batch_size, 3, device=x.device)

            return seg_output

        # Phase 2: Combined training with balanced focus
        else:
            # Primary pass for segmentation
            seg_output = self.base_network(x)

            # Secondary pass for classification (independent gradient flow)
            if hasattr(self.base_network, 'encoder'):
                # Get encoder features
                encoder_features = self.base_network.encoder(x)

                if isinstance(encoder_features, list):
                    bottleneck_features = encoder_features[-1]
                else:
                    bottleneck_features = encoder_features

                self.bottleneck_features = bottleneck_features
                self.last_classification_output = self.classification_head(bottleneck_features)
            else:
                self.last_classification_output = torch.zeros(batch_size, 3, device=x.device)

            return seg_output

    def set_training_phase(self, phase: int):
        """Set the training phase (1: segmentation focus, 2: combined)."""
        assert phase in [1, 2], "Training phase must be 1 or 2"
        self.training_phase = phase
        print(f"[MultitaskUNet] Set training phase to {phase}")

    def enable_fast_inference(self):
        """Enable optimized inference mode."""
        self.inference_mode = True

    def disable_fast_inference(self):
        """Disable optimized inference mode."""
        self.inference_mode = False

    @property
    def deep_supervision(self) -> bool:
        """Get deep supervision status."""
        if hasattr(self.base_network, 'deep_supervision'):
            return self.base_network.deep_supervision
        elif hasattr(self, 'decoder') and hasattr(self.decoder, 'deep_supervision'):
            return self.decoder.deep_supervision
        else:
            return False

    @deep_supervision.setter
    def deep_supervision(self, value: bool):
        """Set deep supervision status."""
        if hasattr(self.base_network, 'deep_supervision'):
            self.base_network.deep_supervision = value
        if hasattr(self, 'decoder') and hasattr(self.decoder, 'deep_supervision'):
            self.decoder.deep_supervision = value

class EarlyStoppingException(Exception):
    """Exception raised to force early stopping in training loop."""
    pass


###############################################################################
# Fixed MultitasknnUNetTrainer with Optimized Training Strategy
###############################################################################
class MultitasknnUNetTrainer(nnUNetTrainer):
    """
    Completely redesigned trainer with improved training stability:
    - Longer segmentation phase for better base performance
    - Carefully calibrated loss weights
    - Balanced class handling for better classification
    - Enhanced monitoring and early stopping
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Initialize parent class
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Training phases ---
        self.segmentation_phase_epochs = 20   # Increased from 5 to 20 - critical fix!
        self.current_phase = 1                # Start in phase 1

        # --- Loss weighting parameters ---
        self.seg_weight = 1.0                  # Base segmentation weight
        self.cls_weight = 0.01                 # Very small weight in phase 1 (changed from 0.05)
        self.phase2_cls_weight = 0.1           # Reduced from 0.2 to 0.1 for phase 2

        # --- Classification parameters ---
        self.num_classes = 3                  # Number of subtypes
        self.case_to_subtype = {}             # Case ID to subtype mapping

        # --- Optimization parameters ---
        self.initial_lr = 1e-4                # Reduced from 3e-4 to 1e-4 for stability
        self.weight_decay = 3e-5              # Adjusted weight decay

        # --- Early stopping parameters ---
        self.best_f1 = 0.0                    # Best classification F1 score
        self.best_dice = 0.0                  # Best segmentation dice score
        self.best_metric = -float('inf')      # Best combined metric
        self.early_stopping_patience = 100    # patience
        self.epochs_without_improvement = 0   # Counter for early stopping
        self.num_epochs = 500                 # Maximum number of epochs

        # --- Class balancing parameters ---
        self.class_counts = torch.tensor([62, 106, 84], dtype=torch.float)  # From dataset stats
        self.use_class_weights = True         # Enable class weighting for classification loss

        # --- Status tracking ---
        self.current_epoch_cls_metrics = None  # Classification metrics for current epoch

        # Performance targets
        self.print_to_log_file("=" * 50)
        self.print_to_log_file("MultitasknnUNetTrainer for pancreatic cancer")
        self.print_to_log_file("=" * 50)
        self.print_to_log_file("Expected performance:")
        self.print_to_log_file("- Whole pancreas DSC: ~0.91+")
        self.print_to_log_file("- Pancreas lesion DSC: ≥0.31")
        self.print_to_log_file("- Classification macro F1: ≥0.7")
        self.print_to_log_file("=" * 50)

    def initialize(self):
        """Initialize the trainer with fixed multitask components."""
        # Initialize parent components
        super().initialize()

        # Log initialization status
        self.print_to_log_file("Initializing MultitasknnUNetTrainer...")

        # Verify dataset paths for debugging
        if hasattr(self, 'preprocessed_dataset_folder_base'):
            self.print_to_log_file(f"Preprocessed dataset base: {self.preprocessed_dataset_folder_base}")
        if hasattr(self, 'preprocessed_dataset_folder'):
            self.print_to_log_file(f"Using preprocessed folder: {self.preprocessed_dataset_folder}")

        # Initialize validation keys for evaluation
        self._initialize_validation_keys()

        # Initialize mapping between case IDs and subtypes
        self._initialize_case_to_subtype_mapping()

        # Debug: Show class distribution
        if self.case_to_subtype:
            subtype_counts = {i: sum(1 for v in self.case_to_subtype.values() if v == i)
                             for i in range(self.num_classes)}
            self.print_to_log_file(f"Class distribution: Subtype 0: {subtype_counts.get(0, 0)}, "
                                  f"Subtype 1: {subtype_counts.get(1, 0)}, "
                                  f"Subtype 2: {subtype_counts.get(2, 0)}")

        # Wrap the base network with our multitask architecture
        original_network = self.network
        self.network = MultitaskUNet(original_network, self.num_classes)
        self.network = self.network.to(self.device)

        # Set training phase
        self.network.set_training_phase(1)  # Start with segmentation phase

        # Create optimizer with improved parameters
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-8
        )

        # Create learning rate scheduler
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        self.lr_scheduler = PolyLRScheduler(
            self.optimizer,
            self.initial_lr,
            self.num_epochs
        )

        # Initialize grad scaler for mixed precision training if using CUDA
        if self.device.type == 'cuda':
            self.grad_scaler = torch.cuda.amp.GradScaler()

        self.print_to_log_file("Multitask initialization complete.")

    def _initialize_validation_keys(self):
        """Initialize validation keys from splits file or derive from folder structure."""
        self.validation_keys = []

        # Try to get validation keys from official splits file
        if hasattr(self, 'preprocessed_dataset_folder_base'):
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            if isfile(splits_file):
                try:
                    self.print_to_log_file(f"Loading validation keys from {splits_file}")
                    splits = load_json(splits_file)

                    if isinstance(splits, list) and len(splits) > self.fold and 'val' in splits[self.fold]:
                        self.validation_keys = splits[self.fold]['val']
                        self.print_to_log_file(f"Loaded {len(self.validation_keys)} validation keys from splits file")
                    else:
                        self.print_to_log_file("Warning: Could not extract validation keys from splits")
                except Exception as e:
                    self.print_to_log_file(f"Error loading splits file: {str(e)}")

        # If no validation keys found, try alternate methods
        if not self.validation_keys:
            # Check if we can find validation keys in the dataset structure
            try:
                # Try examining folder structure
                dataset_folder = None

                # Try to find the dataset folder
                if 'nnUNet_raw' in os.environ:
                    potential_dataset_folder = os.path.join(
                        os.environ['nnUNet_raw'],
                        f"Dataset{self.plans_manager.dataset_name}"
                    )
                    if os.path.exists(potential_dataset_folder):
                        dataset_folder = potential_dataset_folder

                if dataset_folder and os.path.exists(dataset_folder):
                    # Check for validation folder
                    validation_folder = join(dataset_folder, "validation")
                    if os.path.exists(validation_folder):
                        import glob

                        # Find all image files in validation folder
                        image_files = []
                        for subtype_folder in ['subtype0', 'subtype1', 'subtype2']:
                            subtype_path = join(validation_folder, subtype_folder)
                            if os.path.exists(subtype_path):
                                image_files.extend(glob.glob(join(subtype_path, "*.nii.gz")))

                        # Extract case IDs from image files
                        if image_files:
                            val_keys = []
                            for f in image_files:
                                # Remove _0000 suffix and file extension
                                key = os.path.basename(f).replace("_0000.nii.gz", "").replace(".nii.gz", "")
                                if key not in val_keys:
                                    val_keys.append(key)

                            if val_keys:
                                self.validation_keys = val_keys
                                self.print_to_log_file(f"Found {len(val_keys)} validation keys from folder structure")
            except Exception as e:
                self.print_to_log_file(f"Error finding validation keys from folders: {str(e)}")

        # Log validation keys status
        if self.validation_keys:
            examples = self.validation_keys[:5]
            self.print_to_log_file(f"Validation keys examples: {examples}")
        else:
            self.print_to_log_file("Warning: No validation keys found. Validation will be limited.")

    def _initialize_case_to_subtype_mapping(self):
        """Create mapping between case identifiers and cancer subtypes using a simpler approach."""
        self.case_to_subtype = {}


        self.print_to_log_file("Extracting subtypes from case naming patterns")

        # Try to get all available case IDs
        if hasattr(self, 'dataset_tr') and hasattr(self.dataset_tr, 'identifiers'):
            all_case_ids = self.dataset_tr.identifiers
        elif hasattr(self, 'validation_keys'):
            all_case_ids = self.validation_keys
        else:
            all_case_ids = []

            # Extract subtype from name pattern
        for case_id in all_case_ids:
            match = re.search(r'quiz_(\d)_', case_id)
            if match:
                subtype = int(match.group(1))
                if 0 <= subtype <= 2:
                    self.case_to_subtype[case_id] = subtype

        # Log mapping statistics
        if self.case_to_subtype:
            self.print_to_log_file(f"Created {len(self.case_to_subtype)} case-to-subtype mappings")
            examples = list(self.case_to_subtype.items())[:5]
            self.print_to_log_file(f"Mapping examples: {examples}")
        else:
            self.print_to_log_file("WARNING: Failed to create subtype mappings!")

    def on_train_start(self):
        """Additional setup at training start."""
        # Call parent method
        super().on_train_start()

        # Verify our classification mapping using training data
        self.print_to_log_file("Verifying classification label mapping after dataset initialization...")
        self.verify_training_case_mapping()

    def verify_training_case_mapping(self):
        """Verify the class distribution in training data."""
        from collections import defaultdict
        label_count = defaultdict(int)
        all_train_keys = []

        # Try to get training keys from splits
        if hasattr(self, 'preprocessed_dataset_folder_base'):
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            if isfile(splits_file):
                try:
                    splits = load_json(splits_file)
                    if len(splits) > self.fold and 'train' in splits[self.fold]:
                        all_train_keys = splits[self.fold]['train']
                        self.print_to_log_file(f"Found {len(all_train_keys)} training keys from splits_final.json")
                except Exception as e:
                    self.print_to_log_file(f"Error loading training keys from splits: {str(e)}")

        # If no keys found, try from dataset_tr
        if not all_train_keys and hasattr(self, 'dataset_tr') and self.dataset_tr is not None:
            if hasattr(self.dataset_tr, 'identifiers'):
                all_train_keys = self.dataset_tr.identifiers
                self.print_to_log_file(f"Found {len(all_train_keys)} training keys from dataset_tr")

        # Count labels by class
        if all_train_keys:
            for case_id in all_train_keys:
                label = self._get_class_label_for_case(case_id)
                label_count[label] += 1

            # Summarize results
            self.print_to_log_file("==== Training Class Distribution ====")
            self.print_to_log_file(f"Total training keys: {len(all_train_keys)}")
            for lbl, cnt in sorted(label_count.items()):
                self.print_to_log_file(f"Class {lbl}: {cnt} cases ({cnt/len(all_train_keys)*100:.1f}%)")
            self.print_to_log_file("===================================")

            # Update class weights based on actual distribution
            if sum(label_count.values()) > 0:
                self.class_counts = torch.tensor([label_count[0], label_count[1], label_count[2]], dtype=torch.float)
                self.print_to_log_file(f"Updated class counts: {self.class_counts}")

    def on_train_epoch_start(self):
        """Configure phase-specific training parameters at the start of each epoch."""
        super().on_train_epoch_start()

        # Determine and update training phase
        current_epoch = self.current_epoch
        if current_epoch < self.segmentation_phase_epochs:
            if self.current_phase != 1:
                self.current_phase = 1
                self.network.set_training_phase(1)
                self.print_to_log_file("Training Phase 1: Segmentation priority with light classification")
        else:
            if self.current_phase != 2:
                self.current_phase = 2
                self.network.set_training_phase(2)
                self.print_to_log_file("Training Phase 2: Combined segmentation and classification")

        # Log current phase and loss weights
        if self.current_phase == 1:
            self.print_to_log_file(f"Epoch {current_epoch}: Phase 1 - Segmentation priority (seg_weight=1.0, cls_weight={self.cls_weight})")
        else:
            self.print_to_log_file(f"Epoch {current_epoch}: Phase 2 - Combined training (seg_weight=1.0, cls_weight={self.phase2_cls_weight})")

    def _extract_case_id_from_key(self, key):
        """Extract standardized case ID from various key formats."""
        # Handle different input types
        if isinstance(key, (list, tuple, np.ndarray)) and len(key) > 0:
            key = key[0]  # Take first element if it's a sequence

        # Convert to string
        key = str(key)

        # Normalize to canonical form: remove file extensions and _0000 suffix
        key = os.path.basename(key)
        key = key.replace('.npy', '').replace('.nii.gz', '').replace('_0000', '')

        return key

    def _get_class_label_for_case(self, case_id):
        """Determine the class label using a simplified lookup approach."""
        # Normalize the case ID first
        clean_id = self._extract_case_id_from_key(case_id)

        # Direct lookup with normalized ID
        if clean_id in self.case_to_subtype:
            return self.case_to_subtype[clean_id]

        # Fallback pattern matching as last resort
        match = re.search(r'quiz_(\d)_|^(\d)_', clean_id)
        if match:
            # Use the first matching group that has a value
            subtype = int(match.group(1) if match.group(1) is not None else match.group(2))
            if 0 <= subtype <= 2:
                # Cache for future lookups
                self.case_to_subtype[clean_id] = subtype
                return subtype

        # Default to class 0 if nothing matches
        return 0

    def _get_labels_from_case_ids(self, case_ids):
        """Get class labels for a list of case IDs."""
        if not case_ids:
            return torch.tensor([], device=self.device, dtype=torch.long)

        labels = []
        for cid in case_ids:
            label = self._get_class_label_for_case(cid)
            labels.append(label)

        return torch.tensor(labels, device=self.device, dtype=torch.long)


    def train_step(self, batch: dict) -> dict:
        """Enhanced training step with carefully isolated gradient flows."""
        # Unpack batch
        data = batch['data']
        target = batch['target']
        keys = batch.get('keys', None)

        # Get class labels for the batch
        case_ids = []
        if keys is not None:
            if isinstance(keys, (list, tuple)):
                case_ids = [self._extract_case_id_from_key(k) for k in keys]
            elif isinstance(keys, np.ndarray):
                case_ids = [self._extract_case_id_from_key(keys[i]) for i in range(len(keys))]

        cls_target = self._get_labels_from_case_ids(case_ids)

        # Move data to device
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Reset gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Calculate class weights for balancing
        class_weights = None
        if self.use_class_weights and hasattr(self, 'class_counts') and self.class_counts.sum() > 0:
            # Inverse frequency weighting with smoothing
            class_weights = 1.0 / (self.class_counts + 1)
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(self.device)

        # Use automatic mixed precision for faster training on CUDA
        with torch.amp.autocast('cuda', enabled=self.device.type=='cuda'):
            # Forward pass through network
            seg_output = self.network(data)
            cls_output = self.network.last_classification_output

            # Segmentation loss
            seg_loss = self.loss(seg_output, target)

            # Classification loss (only if we have labels)
            if cls_output is not None and len(cls_target) > 0:
                # Apply class weights if available
                if class_weights is not None:
                    cls_loss = F.cross_entropy(cls_output, cls_target, weight=class_weights)
                else:
                    cls_loss = F.cross_entropy(cls_output, cls_target)

                # Combined loss with phase-dependent weighting
                if self.current_phase == 1:
                    loss = seg_loss + self.cls_weight * cls_loss
                else:
                    loss = seg_loss + self.phase2_cls_weight * cls_loss
            else:
                # Fallback if no classification targets
                cls_loss = torch.tensor(0.0, device=self.device)
                loss = seg_loss

        # Backward and optimize with gradient scaling for mixed precision
        if self.device.type == 'cuda' and hasattr(self, 'grad_scaler') and self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        # Return metrics
        return {
            'loss': loss.detach().cpu().numpy(),
            'seg_loss': seg_loss.detach().cpu().numpy(),
            'cls_loss': cls_loss.detach().cpu().numpy()
        }

    def validation_step(self, batch: dict) -> dict:
        """Validation step for both segmentation and classification."""
        # Unpack batch
        data = batch['data']
        target = batch['target']
        keys = batch.get('keys', None)

        # Get class labels for validation
        case_ids = []
        if keys is not None:
            if isinstance(keys, (list, tuple)):
                case_ids = [self._extract_case_id_from_key(k) for k in keys]
            elif isinstance(keys, np.ndarray):
                case_ids = [self._extract_case_id_from_key(keys[i]) for i in range(len(keys))]

        cls_target = self._get_labels_from_case_ids(case_ids)

        # Move data to device
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Forward pass without gradients
        with torch.no_grad():
            # Use automatic mixed precision for faster validation
            with torch.amp.autocast('cuda', enabled=self.device.type=='cuda'):
                # Get segmentation and classification outputs
                seg_output = self.network(data)
                cls_output = self.network.last_classification_output

                # Segmentation loss
                seg_loss = self.loss(seg_output, target)

                # Classification metrics
                cls_metrics = {}
                if cls_output is not None and len(cls_target) > 0:
                    # Classification loss
                    cls_loss = F.cross_entropy(cls_output, cls_target)

                    # Combined loss with phase-dependent weighting
                    if self.current_phase == 1:
                        loss = seg_loss + self.cls_weight * cls_loss
                    else:
                        loss = seg_loss + self.phase2_cls_weight * cls_loss

                    # Classification accuracy
                    _, predicted = torch.max(cls_output.data, 1)
                    accuracy = (predicted == cls_target).float().mean()

                    cls_metrics = {
                        'cls_loss': cls_loss.detach().cpu().numpy(),
                        'cls_acc': accuracy.detach().cpu().numpy(),
                        'cls_pred': predicted.detach().cpu().numpy(),
                        'cls_target': cls_target.detach().cpu().numpy()
                    }
                else:
                    loss = seg_loss
                    cls_metrics = {
                        'cls_loss': np.array(0.0, dtype=np.float32),
                        'cls_acc': np.array(0.0, dtype=np.float32)
                    }

        # Call parent's validation step for segmentation metrics
        seg_metrics = super().validation_step(batch)

        # Combine metrics
        return {**seg_metrics, **cls_metrics, 'loss': loss.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs):
        """Process validation outputs and calculate comprehensive metrics."""
        # Call parent's method for segmentation metrics
        super().on_validation_epoch_end(val_outputs)
        
        # Collect classification predictions and targets
        all_preds = []
        all_targets = []

        for output in val_outputs:
            if 'cls_pred' in output and 'cls_target' in output:
                all_preds.extend(output['cls_pred'].tolist() if hasattr(output['cls_pred'], 'tolist') else [output['cls_pred']])
                all_targets.extend(output['cls_target'].tolist() if hasattr(output['cls_target'], 'tolist') else [output['cls_target']])

            

        # Calculate classification metrics if we have predictions
        if all_preds and all_targets and len(all_preds) == len(all_targets):
            try:
                # Convert to numpy arrays
                all_preds_np = np.array(all_preds)
                all_targets_np = np.array(all_targets)

                # Compute F1 score and accuracy
                f1 = f1_score(all_targets_np, all_preds_np, average='macro')
                accuracy = accuracy_score(all_targets_np, all_preds_np)

                # Compute confusion matrix
                cm = confusion_matrix(all_targets_np, all_preds_np, labels=range(self.num_classes))

                # Update best F1 score
                if f1 > self.best_f1:
                    self.best_f1 = f1

                # Store metrics for this epoch
                self.current_epoch_cls_metrics = {
                    'f1': float(f1),
                    'accuracy': float(accuracy),
                    'confusion_matrix': cm.tolist()
                }

                # Log metrics
                self.print_to_log_file(f"Classification F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
                self.print_to_log_file(f"Confusion Matrix:")
                for i, row in enumerate(cm):
                    self.print_to_log_file(f"   Subtype {i}: {row}")

                # Log F1 score to the logger
                self.initialize_logger()
                self.logger.log('classification_f1', float(f1), self.current_epoch)

            except Exception as e:
                self.print_to_log_file(f"Error calculating classification metrics: {str(e)}")

    def on_epoch_end(self):
        """Enhanced epoch end processing with comprehensive metrics and early stopping."""
        # Call parent's epoch end processing
        super().on_epoch_end()

        # Get current epoch number (already incremented by parent)
        current_epoch_number = self.current_epoch - 1

        # Print detailed summary
        self.print_to_log_file("=" * 50)
        self.print_to_log_file(f"EPOCH {current_epoch_number} SUMMARY:")

        # Get segmentation metrics
        dice_per_class = None
        current_dice = 0
        if hasattr(self, 'logger') and hasattr(self.logger, 'my_fantastic_logging'):
            log_data = self.logger.my_fantastic_logging
            if 'dice_per_class_or_region' in log_data and log_data['dice_per_class_or_region']:
                dice_per_class = log_data['dice_per_class_or_region'][-1]
                if isinstance(dice_per_class, list) and len(dice_per_class) >= 2:
                    whole_pancreas_dice = dice_per_class[0]
                    lesion_dice = dice_per_class[1]
                    self.print_to_log_file(f"Whole pancreas DSC: {whole_pancreas_dice:.4f} / 0.91")
                    self.print_to_log_file(f"Pancreas lesion DSC: {lesion_dice:.4f} / 0.31")
                    current_dice = np.nanmean(dice_per_class)

                    # Track best dice
                    if current_dice > self.best_dice:
                        self.best_dice = current_dice
                else:
                    self.print_to_log_file("No segmentation metrics available yet")
            elif 'mean_fg_dice' in log_data and log_data['mean_fg_dice']:
                current_dice = log_data['mean_fg_dice'][-1]
                self.print_to_log_file(f"Mean foreground DSC: {current_dice:.4f}")

                # Track best dice
                if current_dice > self.best_dice:
                    self.best_dice = current_dice

        # Get classification metrics
        cls_f1 = 0
        if hasattr(self, 'current_epoch_cls_metrics') and self.current_epoch_cls_metrics:
            cls_f1 = self.current_epoch_cls_metrics.get('f1', 0)
            self.print_to_log_file(f"Classification F1: {cls_f1:.4f} (Target: ≥0.7)")

            accuracy = self.current_epoch_cls_metrics.get('accuracy', 0)
            self.print_to_log_file(f"Classification Accuracy: {accuracy:.4f}")

        # Check if requirements are met
        pancreas_req = "✅" if dice_per_class and dice_per_class[0] >= 0.91 else "❌"
        lesion_req = "✅" if dice_per_class and dice_per_class[1] >= 0.31 else "❌"
        cls_req = "✅" if cls_f1 >= 0.7 else "❌"

        self.print_to_log_file(f"Requirements: Pancreas: {pancreas_req} | Lesion: {lesion_req} | Classification: {cls_req}")

        # Calculate combined metric for early stopping
        # Phase-dependent weighting - prioritize segmentation in early phases
        if current_epoch_number < self.segmentation_phase_epochs:
            seg_weight = 0.9
            cls_weight = 0.1
        else:
            # Gradually balance as training progresses
            seg_weight = max(0.5, 0.9 - 0.4 * (current_epoch_number - self.segmentation_phase_epochs) / (100 - self.segmentation_phase_epochs))
            cls_weight = 1.0 - seg_weight

        combined_metric = (current_dice * seg_weight + cls_f1 * cls_weight)
        self.print_to_log_file(f"Combined metric: {combined_metric:.4f} (Seg weight: {seg_weight:.2f}, Cls weight: {cls_weight:.2f})")
        self.print_to_log_file("=" * 50)

        # Check for improvement and early stopping
        if combined_metric > self.best_metric:
            self.best_metric = combined_metric
            self.epochs_without_improvement = 0
            self.print_to_log_file(f"New best combined metric: {combined_metric:.4f}")

            # Save best checkpoint
            if not self.disable_checkpointing:
                self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))
        else:
            self.epochs_without_improvement += 1
            self.print_to_log_file(f"No improvement for {self.epochs_without_improvement} epochs. Best: {self.best_metric:.4f}")

            # Check for early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.print_to_log_file(f"Early stopping triggered after {current_epoch_number} epochs")
                self.num_epochs = current_epoch_number + 1  # Stop after this epoch

                # raise EarlyStoppingException(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement")

        if self.local_rank == 0:
            try:
                self.plot_progress_with_f1()
            except Exception as e:
                self.print_to_log_file(f"Error creating custom progress plot: {str(e)}")

    def initialize_logger(self):
        """Initialize or update the logger to include classification metrics"""
        if not hasattr(self, 'logger') or self.logger is None:
            from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
            self.logger = nnUNetLogger()

        # Add classification_f1 to the logger if not already present
        if 'classification_f1' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['classification_f1'] = []

        return self.logger

    def plot_progress_with_f1(self):
        """Create an enhanced version of the progress plot that includes F1 scores."""
        if self.local_rank == 0:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            from batchgenerators.utilities.file_and_folder_operations import join

            # Make sure we have the logger initialized with F1 scores
            self.initialize_logger()

            # Extract data for plotting
            log_data = self.logger.my_fantastic_logging

            # Ensure classification_f1 exists
            if 'classification_f1' not in log_data:
                log_data['classification_f1'] = []

            # Make sure all lists have the same length
            max_epoch = min([len(i) for i in log_data.values() if len(i) > 0]) - 1
            x_values = list(range(max_epoch + 1))

            # Set up the figure with more subplots
            sns.set(font_scale=1.5)
            fig, ax_all = plt.subplots(4, 1, figsize=(30, 64))

            # Plot 1: Loss and Dice (same as original)
            ax = ax_all[0]
            ax2 = ax.twinx()

            # Loss curves
            ax.plot(x_values, log_data['train_losses'][:max_epoch + 1], color='b', ls='-', label="loss_tr", linewidth=3)
            ax.plot(x_values, log_data['val_losses'][:max_epoch + 1], color='r', ls='-', label="loss_val", linewidth=3)

            # Dice curves
            ax2.plot(x_values, log_data['mean_fg_dice'][:max_epoch + 1], color='g', ls='dotted', label="pseudo dice", linewidth=2)
            ax2.plot(x_values, log_data['ema_fg_dice'][:max_epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)", linewidth=3)

            # Add F1 curve if available
            if len(log_data['classification_f1']) > 0:
                # Pad F1 values if needed
                f1_values = log_data['classification_f1'] + [0] * (max_epoch + 1 - len(log_data['classification_f1']))
                f1_values = f1_values[:max_epoch + 1]
                ax2.plot(x_values, f1_values, color='m', ls='-', label="Classification F1", linewidth=3)

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("metrics")
            ax.legend(loc=(0, 1))
            ax2.legend(loc=(0.2, 1))
            ax.set_title("Loss and Performance Metrics", fontsize=16)

            # Plot 2: Epoch times (same as original)
            ax = ax_all[1]
            ax.plot(x_values, [i - j for i, j in zip(log_data['epoch_end_timestamps'][:max_epoch + 1],
                                                    log_data['epoch_start_timestamps'])][:max_epoch + 1],
                    color='b', ls='-', label="epoch duration", linewidth=3)
            ylim = [0] + [ax.get_ylim()[1]]
            ax.set(ylim=ylim)
            ax.set_xlabel("epoch")
            ax.set_ylabel("time [s]")
            ax.legend(loc=(0, 1))
            ax.set_title("Epoch Duration", fontsize=16)

            # Plot 3: Learning rate (same as original)
            ax = ax_all[2]
            ax.plot(x_values, log_data['lrs'][:max_epoch + 1], color='b', ls='-', label="learning rate", linewidth=3)
            ax.set_xlabel("epoch")
            ax.set_ylabel("learning rate")
            ax.legend(loc=(0, 1))
            ax.set_title("Learning Rate Schedule", fontsize=16)

            # Plot 4: NEW! Classification F1 Score on its own plot
            ax = ax_all[3]
            if len(log_data['classification_f1']) > 0:
                # Pad F1 values if needed
                f1_values = log_data['classification_f1'] + [0] * (max_epoch + 1 - len(log_data['classification_f1']))
                f1_values = f1_values[:max_epoch + 1]
                ax.plot(x_values, f1_values, color='m', ls='-', label="F1 Score", linewidth=3)

                # Add target line at 0.7
                ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label="Target F1 (0.7)")

                # Set range from 0 to 1
                ax.set_ylim([0, 1])

                ax.set_xlabel("epoch")
                ax.set_ylabel("F1 Score")
                ax.legend(loc=(0, 1))
                ax.set_title("Classification Performance (F1 Score)", fontsize=16)

            plt.tight_layout()

            # Save standard progress plot
            fig.savefig(join(self.output_folder, "progress.png"))

            # Also save a dedicated classification plot
            fig_cls = plt.figure(figsize=(15, 10))
            ax_cls = fig_cls.add_subplot(111)

            if len(log_data['classification_f1']) > 0:
                # Plot F1 score
                ax_cls.plot(x_values[:len(log_data['classification_f1'])],
                            log_data['classification_f1'],
                            color='m', ls='-', marker='o', label="F1 Score", linewidth=2)

                # Add target line
                ax_cls.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label="Target F1 (0.7)")

                # Set range from 0 to 1
                ax_cls.set_ylim([0, 1])

                ax_cls.set_xlabel("Epoch", fontsize=12)
                ax_cls.set_ylabel("F1 Score", fontsize=12)
                ax_cls.legend(loc='best')
                ax_cls.set_title("Classification Performance Over Time", fontsize=14)
                ax_cls.grid(True, alpha=0.3)

                plt.tight_layout()
                fig_cls.savefig(join(self.output_folder, "classification_f1.png"))

            plt.close()

    def save_checkpoint(self, filename: str) -> None:
        """Enhanced checkpoint saving with classification metrics."""
        # Call parent's checkpoint saving
        super().save_checkpoint(filename)

        # Save additional classification metrics
        metrics_file = filename.replace('.pth', '_metrics.json')
        try:
            # Convert case_to_subtype keys to strings for JSON compatibility
            serializable_mapping = {str(k): int(v) for k, v in self.case_to_subtype.items()}

            # Save metrics
            save_json({
                'best_f1': float(self.best_f1),
                'best_dice': float(self.best_dice),
                'best_metric': float(self.best_metric),
                'epochs_without_improvement': self.epochs_without_improvement,
                'training_phase': self.current_phase,
                'class_counts': self.class_counts.tolist() if hasattr(self, 'class_counts') else None
            }, metrics_file)
        except Exception as e:
            self.print_to_log_file(f"Error saving classification metrics: {str(e)}")

    def load_checkpoint(self, filename: str, train: bool = True) -> None:
        """Enhanced checkpoint loading with classification metrics."""
        # Call parent's checkpoint loading
        super().load_checkpoint(filename, train)

        # Load additional classification metrics
        metrics_file = filename.replace('.pth', '_metrics.json')
        if isfile(metrics_file):
            try:
                metrics = load_json(metrics_file)
                self.best_f1 = metrics.get('best_f1', 0.0)
                self.best_dice = metrics.get('best_dice', 0.0)
                self.best_metric = metrics.get('best_metric', -float('inf'))
                self.epochs_without_improvement = metrics.get('epochs_without_improvement', 0)
                self.current_phase = metrics.get('training_phase', 2)

                if 'class_counts' in metrics and metrics['class_counts']:
                    self.class_counts = torch.tensor(metrics['class_counts'], dtype=torch.float)

                if hasattr(self, 'network') and hasattr(self.network, 'set_training_phase'):
                    self.network.set_training_phase(self.current_phase)

                self.print_to_log_file(f"Restored classification metrics from {metrics_file}")
            except Exception as e:
                self.print_to_log_file(f"Error loading classification metrics: {str(e)}")

    def predict_from_files(self, image_files: List[str], output_files: List[str],
                           save_probabilities: bool = False, overwrite: bool = True,
                           num_threads_preprocessing: int = 1,
                           num_threads_postprocessing: int = 1):
        """Predict and save segmentation and classification results from image files."""
        # Enable fast inference mode if available
        if hasattr(self.network, 'enable_fast_inference'):
            self.network.enable_fast_inference()

        try:
            # Run segmentation prediction first
            super().predict_from_files(
                image_files, output_files, save_probabilities, overwrite,
                num_threads_preprocessing, num_threads_postprocessing
            )

            # Extract case identifiers for classification
            case_ids = []
            for img_file in image_files:
                # Extract case ID from file path
                filename = os.path.basename(img_file)
                case_id = filename.replace('_0000.nii.gz', '').replace('.nii.gz', '')
                case_ids.append(case_id)

            # Generate classification output folder
            output_folder = os.path.dirname(output_files[0])

            # Run classification
            self._predict_classification(output_folder, case_ids)

        finally:
            # Disable fast inference mode
            if hasattr(self.network, 'disable_fast_inference'):
                self.network.disable_fast_inference()

    def _predict_classification(self, output_folder: str, case_identifiers: List[str]) -> None:
        """Run classification prediction and save results to CSV."""
        import csv

        # Log start of classification prediction
        self.print_to_log_file(f"Running classification prediction for {len(case_identifiers)} cases...")

        # Initialize containers for classification results
        cls_results = []

        # Set network to evaluation mode
        self.network.eval()

        # Process each case
        for case_id in case_identifiers:
            try:
                # For each case, find input file
                case_data = None

                # Try direct NPZ file loading from preprocessed folder
                npz_file = join(self.preprocessed_dataset_folder, f"{case_id}.npz")
                if isfile(npz_file):
                    case_data = np.load(npz_file)['data']

                # Try NPY file from preprocessed folder
                npy_file = join(self.preprocessed_dataset_folder, f"{case_id}.npy")
                if case_data is None and isfile(npy_file):
                    case_data = np.load(npy_file)

                # Try other locations
                if case_data is None:
                    for part in ['0000', '_0000', '']:
                        for ext in ['.npz', '.npy']:
                            test_file = join(self.preprocessed_dataset_folder, f"{case_id}{part}{ext}")
                            if isfile(test_file):
                                try:
                                    if ext == '.npz':
                                        case_data = np.load(test_file)['data']
                                    else:
                                        case_data = np.load(test_file)
                                    break
                                except:
                                    continue
                        if case_data is not None:
                            break

                # If data found, run prediction
                if case_data is not None:
                    with torch.no_grad():
                        # Convert to tensor
                        case_data = torch.from_numpy(case_data).to(self.device)

                        # Ensure correct input shape (batch dimension)
                        if len(case_data.shape) == 3:
                            case_data = case_data[None, None]
                        elif len(case_data.shape) == 4:
                            case_data = case_data[None]

                        # Forward pass
                        with torch.amp.autocast('cuda', enabled=self.device.type=='cuda'):
                            _ = self.network(case_data)
                            cls_output = self.network.last_classification_output

                        # Get prediction
                        if cls_output is not None:
                            probs = F.softmax(cls_output, dim=1)
                            _, predicted = torch.max(probs, dim=1)
                            prediction = predicted.item()

                            cls_results.append((f"{case_id}.nii.gz", prediction))
                        else:
                            # No classification output
                            cls_results.append((f"{case_id}.nii.gz", 0))
                else:
                    # No data found, default to class 0
                    self.print_to_log_file(f"Warning: Could not find data for case {case_id}")
                    cls_results.append((f"{case_id}.nii.gz", 0))

            except Exception as e:
                # Handle errors gracefully
                self.print_to_log_file(f"Error classifying case {case_id}: {str(e)}")
                cls_results.append((f"{case_id}.nii.gz", 0))

        # Save classification results to CSV
        csv_path = join(output_folder, "subtype_results.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Names', 'Subtype'])
            for case_file, subtype in cls_results:
                writer.writerow([case_file, subtype])

        self.print_to_log_file(f"Classification results saved to {csv_path}")