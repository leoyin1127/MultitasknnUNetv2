#!/usr/bin/env python
"""
Enhanced MultitasknnUNetTrainer for pancreatic cancer segmentation and classification.
This implementation includes fixes for the observed training issues:
- Separate gradient flow paths to prevent task interference
- Two-phase training strategy (segmentation first, then combined)
- Simplified classification head with balanced loss contributions
- Optimized feedback during training with detailed metrics
"""

# Suppress TorchDynamo errors to avoid in-place broadcasting issues
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
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import (
    join, load_json, save_json, isfile, maybe_mkdir_p
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

# Use public dataloading modules
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder

import SimpleITK as sitk

###############################################################################
# Simple Classification Head Implementation
###############################################################################
class SimpleClassificationHead(nn.Module):
    """
    Simplified classification head with minimal interference to segmentation.
    Uses global pooling to avoid introducing spatial biases.
    """
    def __init__(self, in_channels: int, num_classes: int = 3, dropout_rate: float = 0.5):
        super(SimpleClassificationHead, self).__init__()
        
        # Global average pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Simple MLP classifier with regularization
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            x: Input feature tensor from the encoder
            
        Returns:
            Classification logits
        """
        # Global average pooling
        x = self.gap(x).view(x.size(0), -1)
        
        # Pass through classifier
        return self.classifier(x)

###############################################################################
# MultitaskUNet with Separate Paths for Classification and Segmentation
###############################################################################
class MultitaskUNet(nn.Module):
    """
    Enhanced UNet architecture with separate paths for segmentation and classification
    to reduce task interference.
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
                # Get channels from the last encoder stage
                bottleneck_dim = encoder.stages[-1].output_channels
                print(f"[MultitaskUNet] Detected bottleneck dimension: {bottleneck_dim}")
        
        # Create a simple classification head
        self.classification_head = SimpleClassificationHead(bottleneck_dim, num_classes)
        
        # Store reference to decoder and encoder for convenience
        if hasattr(base_network, 'decoder'):
            self.decoder = base_network.decoder
        if hasattr(base_network, 'encoder'):
            self.encoder = base_network.encoder
        
        # For storing intermediate outputs
        self.last_classification_output = None
        self.encoder_features = None
        
        # Training vs inference mode
        self.training_phase = 1  # 1: segmentation only, 2: combined
        self.inference_mode = False
        
        # Print network architecture summary
        print(f"[MultitaskUNet] Initialized with segmentation network and classification head")
        print(f"[MultitaskUNet] Classification head input dimensions: {bottleneck_dim}")
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass handling both segmentation and classification tasks.
        Uses separate paths to reduce interference during training.
        
        Args:
            x: Input image tensor
            
        Returns:
            Segmentation output (classification stored in self.last_classification_output)
        """
        batch_size = x.shape[0]
        
        # Store input tensor shape for debugging
        input_shape = x.shape
        
        # Forward path varies based on training phase
        if self.training and self.training_phase == 1:
            # Phase 1: Segmentation only (classification head detached)
            seg_output = self.base_network(x)
            
            # Get encoder features for classification, but detach to prevent gradient flow
            if hasattr(self.base_network, 'encoder'):
                # For networks with explicit encoder-decoder structure
                encoder_features = self.base_network.encoder(x)
                
                # Handle different encoder output formats
                if isinstance(encoder_features, list):
                    bottleneck_features = encoder_features[-1].detach()  # Last encoder level
                else:
                    bottleneck_features = encoder_features.detach()
                
                # Run classification on detached features
                self.last_classification_output = self.classification_head(bottleneck_features)
            else:
                # Fallback for other network architectures
                # Set classification output to zeros to avoid errors
                self.last_classification_output = torch.zeros(batch_size, 3, device=x.device)
                
            return seg_output
            
        elif self.inference_mode or not self.training:
            # Inference mode: run full network once for efficiency
            seg_output = self.base_network(x)
            
            # Get encoder features for classification
            if hasattr(self.base_network, 'encoder'):
                # For networks with explicit encoder-decoder structure
                encoder_features = self.base_network.encoder(x)
                
                # Handle different encoder output formats
                if isinstance(encoder_features, list):
                    bottleneck_features = encoder_features[-1]  # Last encoder level
                else:
                    bottleneck_features = encoder_features
                
                # Run classification
                self.last_classification_output = self.classification_head(bottleneck_features)
            else:
                # Fallback for other network architectures
                # Set classification output to zeros to avoid errors
                self.last_classification_output = torch.zeros(batch_size, 3, device=x.device)
                
            return seg_output
            
        else:
            # Phase 2: Combined training with both tasks
            # First run segmentation
            seg_output = self.base_network(x)
            
            # Then run encoder again for classification (shared weights but separate computation)
            if hasattr(self.base_network, 'encoder'):
                # For networks with explicit encoder-decoder structure
                encoder_features = self.base_network.encoder(x)
                
                # Handle different encoder output formats
                if isinstance(encoder_features, list):
                    bottleneck_features = encoder_features[-1]  # Last encoder level
                else:
                    bottleneck_features = encoder_features
                
                # Run classification
                self.last_classification_output = self.classification_head(bottleneck_features)
            else:
                # Fallback for other network architectures
                # Set classification output to zeros to avoid errors
                self.last_classification_output = torch.zeros(batch_size, 3, device=x.device)
                
            return seg_output

    def set_training_phase(self, phase: int):
        """
        Set the training phase:
        - Phase 1: Segmentation only
        - Phase 2: Combined segmentation and classification
        
        Args:
            phase: Training phase (1 or 2)
        """
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

###############################################################################
# Fixed MultitasknnUNetTrainer with Phase-based Training Strategy
###############################################################################
class MultitasknnUNetTrainer(nnUNetTrainer):
    """
    Fixed MultitasknnUNetTrainer for pancreatic cancer segmentation and classification.
    Implements a two-phase training approach to avoid task interference:
    1. Initial phase: Focus on segmentation task
    2. Combined phase: Train both tasks with balanced loss contributions
    
    Key improvements:
    - Simplified architecture with separate paths for tasks
    - Careful gradient handling to prevent interference
    - Enhanced logging and monitoring
    - Optimized inference with combined output
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Initialize parent class
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # --- Training phases ---
        self.segmentation_phase_epochs = 10   # Phase 1: Focus on segmentation
        self.current_phase = 1                # Start in phase 1
        
        # --- Loss weighting parameters ---
        self.seg_weight = 1.0                 # Keep segmentation weight fixed
        self.cls_weight = 0.2                 # Lower classification weight to avoid interference
        
        # --- Classification parameters ---
        self.num_classes = 3                  # Number of subtypes
        self.case_to_subtype = {}             # Case ID to subtype mapping
        
        # --- Optimization parameters ---
        self.initial_lr = 3e-4                # Lower initial learning rate for stability
        self.weight_decay = 1e-4              # Slightly stronger regularization
        
        # --- Early stopping parameters ---
        self.best_f1 = 0.0                    # Best classification F1 score
        self.best_metric = -float('inf')      # Best combined metric
        self.early_stopping_patience = 25     # Epochs without improvement before stopping
        self.epochs_without_improvement = 0    # Counter for early stopping
        self.num_epochs = 250                 # Maximum number of epochs
        
        # --- Status tracking ---
        self.current_epoch_cls_metrics = None  # Classification metrics for current epoch
        
        # Welcome message and performance targets
        self.print_to_log_file("=" * 50)
        self.print_to_log_file("MultitasknnUNetTrainer (Fixed Version) for pancreatic cancer")
        self.print_to_log_file("=" * 50)
        self.print_to_log_file("Expected performance:")
        self.print_to_log_file("- Whole pancreas DSC: ~0.91+")
        self.print_to_log_file("- Pancreas lesion DSC: ≥0.31")
        self.print_to_log_file("- Classification macro F1: ≥0.7")
        self.print_to_log_file("=" * 50)
        
    def initialize(self):
        """
        Initialize the trainer with fixed multitask components.
        Implements two-phase training strategy for better stability.
        """
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
        
        # Create optimizer with appropriate learning rate and weight decay
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
        """
        Initialize validation keys from splits file or derive from folder structure.
        """
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
        """
        Create mapping between case identifiers and cancer subtypes.
        Uses pattern matching to extract subtype information from filenames.
        """
        self.case_to_subtype = {}
        
        # Try to find mapping from folder structure
        try:
            # Try to find the dataset folder
            dataset_folder = None
            
            if 'nnUNet_raw' in os.environ:
                potential_dataset_folder = os.path.join(
                    os.environ['nnUNet_raw'], 
                    f"Dataset{self.plans_manager.dataset_name}"
                )
                if os.path.exists(potential_dataset_folder):
                    dataset_folder = potential_dataset_folder
            
            if dataset_folder and os.path.exists(dataset_folder):
                self.print_to_log_file(f"Looking for subtype information in {dataset_folder}")
                
                import glob
                
                # Check for train and validation folders
                subtypes_found = {0: 0, 1: 0, 2: 0}
                
                for location in ['train', 'validation']:
                    location_path = join(dataset_folder, location)
                    
                    if os.path.exists(location_path):
                        for subtype_id in range(3):
                            subtype_folder = join(location_path, f"subtype{subtype_id}")
                            
                            if os.path.exists(subtype_folder):
                                self.print_to_log_file(f"Processing {location}/subtype{subtype_id}")
                                
                                # Find all image files
                                image_files = glob.glob(join(subtype_folder, "*.nii.gz"))
                                
                                for file_path in image_files:
                                    filename = os.path.basename(file_path)
                                    # Store several variants of the case ID
                                    case_id = filename.replace("_0000.nii.gz", "").replace(".nii.gz", "")
                                    
                                    # Add to mapping
                                    self.case_to_subtype[case_id] = subtype_id
                                    self.case_to_subtype[f"{case_id}.nii.gz"] = subtype_id
                                    self.case_to_subtype[f"{case_id}_0000.nii.gz"] = subtype_id
                                    
                                    # Add different variations
                                    if "quiz_" in case_id:
                                        clean_id = case_id.replace("quiz_", "")
                                        if "_" in clean_id:
                                            subtype_str, case_num = clean_id.split("_", 1)
                                            # If the pattern matches quiz_0_123, store also key 0_123
                                            if subtype_str.isdigit() and int(subtype_str) == subtype_id:
                                                self.case_to_subtype[f"{subtype_str}_{case_num}"] = subtype_id
                                
                                subtypes_found[subtype_id] += len(image_files)
                
                # Log what we found                
                self.print_to_log_file(
                    f"Extracted from folders: Subtype 0: {subtypes_found[0]}, "
                    f"Subtype 1: {subtypes_found[1]}, Subtype 2: {subtypes_found[2]} cases"
                )
        except Exception as e:
            self.print_to_log_file(f"Error extracting subtypes from folders: {str(e)}")
        
        # If mapping is still empty, try pattern matching on case identifiers
        if not self.case_to_subtype and hasattr(self, 'validation_keys') and self.validation_keys:
            self.print_to_log_file("Attempting to extract subtype info from validation keys...")
            
            for key in self.validation_keys:
                # Try to extract subtype from key name using patterns
                for pattern in [r'quiz_(\d)_\d+', r'(\d)_\d+']:
                    match = re.search(pattern, key)
                    if match:
                        subtype = int(match.group(1))
                        if 0 <= subtype <= 2:
                            self.case_to_subtype[key] = subtype
                            break
        
        # Fallback: Try to infer from key names as last resort
        if not self.case_to_subtype:
            self.print_to_log_file("WARNING: Could not extract subtypes from structure. Trying to infer from key names...")
            
            # Try to get all keys from dataset
            dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            try:
                all_keys = dataset_class.get_identifiers(self.preprocessed_dataset_folder)
                
                pattern_matched = 0
                for key in all_keys:
                    for pattern in [r'quiz_(\d)_', r'(\d)_']:
                        match = re.search(pattern, key)
                        if match:
                            subtype = int(match.group(1))
                            if 0 <= subtype <= 2:
                                self.case_to_subtype[key] = subtype
                                pattern_matched += 1
                                break
                
                if pattern_matched > 0:
                    self.print_to_log_file(f"Inferred {pattern_matched} subtype mappings from key patterns")
            except Exception as e:
                self.print_to_log_file(f"Error inferring subtypes: {str(e)}")
        
        # Log examples of the mapping
        if self.case_to_subtype:
            examples = list(self.case_to_subtype.items())[:5]
            self.print_to_log_file(f"Case-to-subtype mapping examples: {examples}")
        else:
            self.print_to_log_file("WARNING: Could not create subtype mapping. Classification will not work properly!")

    def on_train_epoch_start(self):
        """
        Configure phase-specific training parameters at the start of each epoch.
        """
        super().on_train_epoch_start()
        
        # Determine and update training phase
        current_epoch = self.current_epoch
        if current_epoch < self.segmentation_phase_epochs:
            if self.current_phase != 1:
                self.current_phase = 1
                self.network.set_training_phase(1)
                self.print_to_log_file("Training Phase 1: Focusing on segmentation")
        else:
            if self.current_phase != 2:
                self.current_phase = 2
                self.network.set_training_phase(2)
                self.print_to_log_file("Training Phase 2: Combined segmentation and classification")
        
        # Log current phase and loss weights
        if self.current_phase == 1:
            self.print_to_log_file(f"Epoch {current_epoch}: Phase 1 - Segmentation focus (seg_weight=1.0, cls_weight=0.0)")
        else:
            self.print_to_log_file(f"Epoch {current_epoch}: Phase 2 - Combined training (seg_weight=1.0, cls_weight={self.cls_weight})")

    def _extract_case_id_from_key(self, key):
        """
        Extract standardized case ID from various key formats.
        
        Args:
            key: Raw case identifier (can be string, list, array, etc.)
            
        Returns:
            Standardized case ID string
        """
        # Handle different input types
        if isinstance(key, (list, tuple, np.ndarray)) and len(key) > 0:
            key = key[0]  # Take first element if it's a sequence
            
        # Convert to string if it's not already
        key = str(key)
        
        # Normalize the key format
        key = os.path.basename(key)
        key = key.replace('.npy', '').replace('.nii.gz', '').replace('_0000', '')
        
        # Try multiple extraction patterns
        patterns = [
            (r'quiz_(\d+)_(\d+)', lambda m: f"quiz_{m.group(1)}_{m.group(2)}"),
            (r'quiz_(\d+)', lambda m: f"quiz_{m.group(1)}"),
            (r'(\d+)_(\d+)', lambda m: f"{m.group(1)}_{m.group(2)}")
        ]
        
        for pattern, formatter in patterns:
            match = re.search(pattern, key)
            if match:
                try:
                    return formatter(match)
                except Exception:
                    pass
        
        # If no pattern matched, return normalized key
        return key

    def _get_labels_from_case_ids(self, case_ids):
        """
        Get class labels for a list of case IDs.
        
        Args:
            case_ids: List of case identifiers
            
        Returns:
            Tensor of class labels
        """
        if not case_ids:
            return torch.tensor([], device=self.device, dtype=torch.long)
            
        labels = []
        for cid in case_ids:
            label = self._get_class_label_for_case(cid)
            labels.append(label)
        
        return torch.tensor(labels, device=self.device, dtype=torch.long)

    def _get_class_label_for_case(self, case_id):
        """
        Determine the class label for a case ID using multiple lookup strategies.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Class label (0, 1, or 2)
        """
        # Try direct lookup first
        if case_id in self.case_to_subtype:
            return self.case_to_subtype[case_id]
        
        # Try with standardized key format
        clean_id = self._extract_case_id_from_key(case_id)
        if clean_id in self.case_to_subtype:
            # Cache for future lookups
            self.case_to_subtype[case_id] = self.case_to_subtype[clean_id]
            return self.case_to_subtype[clean_id]
        
        # Try pattern matching if still not found
        for pattern in [r'quiz_(\d)_', r'_(\d)_', r'^(\d)_']:
            match = re.search(pattern, str(case_id))
            if match:
                subtype = int(match.group(1))
                if 0 <= subtype <= 2:
                    # Cache for future lookups
                    self.case_to_subtype[case_id] = subtype
                    return subtype
        
        # Default to class 0 if nothing matches
        return 0

    def train_step(self, batch: dict) -> dict:
        """
        Enhanced training step with phase-based approach to prevent interference.
        
        Args:
            batch: Data batch containing images, targets, and keys
            
        Returns:
            Dictionary with loss values and metrics
        """
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
        
        # Use automatic mixed precision for faster training on CUDA
        with torch.amp.autocast('cuda', enabled=self.device.type=='cuda'):
            # Phase 1: Segmentation focus
            if self.current_phase == 1:
                # Forward pass through segmentation network
                seg_output = self.network(data)
                
                # Segmentation loss
                seg_loss = self.loss(seg_output, target)
                
                # Backward pass with segmentation loss only
                loss = seg_loss
                
                # Set placeholder for classification loss
                cls_loss = torch.tensor(0.0, device=self.device)
            
            # Phase 2: Combined training
            else:
                # Forward pass through network (gets both segmentation and classification)
                seg_output = self.network(data)
                cls_output = self.network.last_classification_output
                
                # Segmentation loss
                seg_loss = self.loss(seg_output, target)
                
                # Classification loss (only if we have labels)
                if cls_output is not None and len(cls_target) > 0:
                    cls_loss = F.cross_entropy(cls_output, cls_target)
                    
                    # Combined loss with fixed weighting
                    loss = seg_loss + self.cls_weight * cls_loss
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
        """
        Validation step for both segmentation and classification.
        
        Args:
            batch: Data batch containing images, targets, and keys
            
        Returns:
            Dictionary with validation metrics
        """
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
                        loss = seg_loss
                    else:
                        loss = seg_loss + self.cls_weight * cls_loss
                    
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
        """
        Process validation outputs and calculate comprehensive metrics.
        
        Args:
            val_outputs: List of outputs from validation steps
        """
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
                
            except Exception as e:
                self.print_to_log_file(f"Error calculating classification metrics: {str(e)}")

    def on_epoch_end(self):
        """
        Enhanced epoch end processing with comprehensive metrics and early stopping.
        """
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
                    self.print_to_log_file(f"Whole pancreas DSC: {dice_per_class[0]:.4f} / 0.91")
                    self.print_to_log_file(f"Pancreas lesion DSC: {dice_per_class[1]:.4f} / 0.31")
                    current_dice = np.nanmean(dice_per_class)
                else:
                    self.print_to_log_file("No segmentation metrics available yet")
            elif 'mean_fg_dice' in log_data and log_data['mean_fg_dice']:
                current_dice = log_data['mean_fg_dice'][-1]
                self.print_to_log_file(f"Mean foreground DSC: {current_dice:.4f}")
        
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
        # Weight segmentation more than classification initially, then balance over time
        seg_weight = 0.7 if current_epoch_number < self.segmentation_phase_epochs else 0.5
        cls_weight = 1.0 - seg_weight
        
        combined_metric = (current_dice * seg_weight + cls_f1 * cls_weight)
        self.print_to_log_file(f"Combined metric: {combined_metric:.4f}")
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

    def save_checkpoint(self, filename: str) -> None:
        """
        Enhanced checkpoint saving with classification metrics.
        """
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
                'best_metric': float(self.best_metric),
                'epochs_without_improvement': self.epochs_without_improvement,
                'training_phase': self.current_phase,
                'case_to_subtype_examples': {k: v for i, (k, v) in enumerate(serializable_mapping.items()) if i < 20}
            }, metrics_file)
        except Exception as e:
            self.print_to_log_file(f"Error saving classification metrics: {str(e)}")

    def load_checkpoint(self, filename: str, train: bool = True) -> None:
        """
        Enhanced checkpoint loading with classification metrics.
        """
        # Call parent's checkpoint loading
        super().load_checkpoint(filename, train)
        
        # Load additional classification metrics
        metrics_file = filename.replace('.pth', '_metrics.json')
        if isfile(metrics_file):
            try:
                metrics = load_json(metrics_file)
                self.best_f1 = metrics.get('best_f1', 0.0)
                self.best_metric = metrics.get('best_metric', -float('inf'))
                self.epochs_without_improvement = metrics.get('epochs_without_improvement', 0)
                self.current_phase = metrics.get('training_phase', 2)
                
                if hasattr(self, 'network') and hasattr(self.network, 'set_training_phase'):
                    self.network.set_training_phase(self.current_phase)
                
                self.print_to_log_file(f"Restored classification metrics from {metrics_file}")
            except Exception as e:
                self.print_to_log_file(f"Error loading classification metrics: {str(e)}")

    def predict_from_files(self, image_files: List[str], output_files: List[str], 
                           save_probabilities: bool = False, overwrite: bool = True,
                           num_threads_preprocessing: int = 1,
                           num_threads_postprocessing: int = 1):
        """
        Predict and save segmentation and classification results from image files.
        """
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
        """
        Run classification prediction and save results to CSV.
        """
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