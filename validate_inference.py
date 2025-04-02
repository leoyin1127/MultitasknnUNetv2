#!/usr/bin/env python3
"""
Improved inference script for pancreatic cancer segmentation and classification
with robust handling of shape mismatches
"""

import os
import time
import numpy as np
import torch
import csv
import sys
import glob
import traceback
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json, isfile

# First, properly disable torch dynamo BEFORE any other imports
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Additional import to handle spatial mismatches
import torch.nn.functional as F

def improved_inference(
    task_id=900,
    test_folder="/content/dataset/test",
    output_folder="./inference_results",
    fold=0,
    checkpoint_name="checkpoint_best.pth"
):
    """
    Improved inference implementation with proper shape handling
    """
    print("\n========== IMPROVED INFERENCE PIPELINE ==========")
    print(f"Test folder: {test_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Using fold: {fold}")
    print(f"Checkpoint: {checkpoint_name}")
    print("=================================================\n")

    start_time = time.time()
    maybe_mkdir_p(output_folder)

    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    # Import custom trainer module
    try: 
        from multitask_trainer import MultitasknnUNetTrainer
        print("Successfully imported MultitasknnUNetTrainer")
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Make sure multitask_trainer.py is in the current directory.")
        return False

    # Setup paths
    nnunet_results = os.environ.get("nnUNet_results", "/content/output/nnUNet_results")
    model_folder = join(nnunet_results, f"Dataset{task_id:03d}_PancreasCancer/MultitasknnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
    fold_dir = join(model_folder, f"fold_{fold}")
    checkpoint_path = join(fold_dir, checkpoint_name)

    # Check for checkpoint
    if not isfile(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found.")
        alternate_checkpoint = join(fold_dir, "checkpoint_final.pth")
        if isfile(alternate_checkpoint):
            checkpoint_path = alternate_checkpoint
            print(f"Using alternate checkpoint: {checkpoint_path}")
        else:
            print(f"Error: No valid checkpoint found in {fold_dir}. Cannot proceed.")
            return False
    else:
        print(f"Using specified checkpoint: {checkpoint_path}")

    # Find test files
    input_files = []
    case_ids = []
    print(f"Searching for test files in: {test_folder}")
    
    if os.path.isdir(test_folder):
        # First check if there are subtype folders
        subtype_dirs = [d for d in os.listdir(test_folder) if os.path.isdir(join(test_folder, d)) and 'subtype' in d]
        
        if subtype_dirs:
            print(f"Found {len(subtype_dirs)} subtype directories")
            for subtype_dir in subtype_dirs:
                subtype_path = join(test_folder, subtype_dir)
                for file in glob.glob(join(subtype_path, "*_0000.nii.gz")):
                    base_name = os.path.basename(file)
                    case_id = base_name.replace("_0000.nii.gz", "")
                    input_files.append(file)
                    case_ids.append(case_id)
        else:
            # No subtype folders, look directly in the test folder
            for file in glob.glob(join(test_folder, "*_0000.nii.gz")):
                base_name = os.path.basename(file)
                case_id = base_name.replace("_0000.nii.gz", "")
                input_files.append(file)
                case_ids.append(case_id)
            
            # Also check for files without the _0000 suffix
            for file in glob.glob(join(test_folder, "*.nii.gz")):
                if "_0000" not in file:
                    base_name = os.path.basename(file)
                    case_id = base_name.replace(".nii.gz", "")
                    if case_id not in case_ids:  # Avoid duplicates
                        input_files.append(file)
                        case_ids.append(case_id)

    if not input_files:
        print(f"Error: No test files found in {test_folder}")
        return False

    print(f"Found {len(input_files)} test files")

    # Set up model and initialization
    try:
        # Load configuration files
        plans_file = join(model_folder, "plans.json")
        dataset_json_file = join(model_folder, "dataset.json")
        
        # Fallback paths if files not found in model folder
        if not isfile(plans_file):
            nnunet_preprocessed = os.environ.get("nnUNet_preprocessed", "/content/output/nnUNet_preprocessed")
            plans_file_preprocessed = join(nnunet_preprocessed, f"Dataset{task_id:03d}_PancreasCancer/nnUNetResEncUNetMPlans.json")
            if isfile(plans_file_preprocessed):
                plans_file = plans_file_preprocessed
        
        if not isfile(dataset_json_file):
            nnunet_raw = os.environ.get("nnUNet_raw", "/content/output/nnUNet_raw")
            dataset_json_file_raw = join(nnunet_raw, f"Dataset{task_id:03d}_PancreasCancer/dataset.json")
            if isfile(dataset_json_file_raw):
                dataset_json_file = dataset_json_file_raw
        
        if not isfile(plans_file) or not isfile(dataset_json_file):
            print(f"Error: Required configuration files not found:")
            print(f"  Plans file: {plans_file} - Exists: {isfile(plans_file)}")
            print(f"  Dataset file: {dataset_json_file} - Exists: {isfile(dataset_json_file)}")
            return False

        # Load configurations
        print(f"Loading plans from: {plans_file}")
        print(f"Loading dataset from: {dataset_json_file}")
        plans = load_json(plans_file)
        dataset_json = load_json(dataset_json_file)

        # Import necessary components from nnUNetv2
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
        from nnunetv2.utilities.label_handling.label_handling import LabelManager
        from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        # Create managers
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration("3d_fullres")
        label_manager = plans_manager.get_label_manager(dataset_json)

        # Create image reader/writer - Pass the loaded dataset_json, not the file path
        reader_writer_class = determine_reader_writer_from_dataset_json(dataset_json)
        reader_writer = reader_writer_class()

        # Initialize nnUNetPredictor which has robust error handling
        print("Initializing nnUNetPredictor...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        # Create and initialize trainer to load the model
        print("Loading model from checkpoint...")
        trainer = MultitasknnUNetTrainer(plans=plans, configuration="3d_fullres", fold=fold, dataset_json=dataset_json, device=device)
        trainer.initialize()
        
        # Load checkpoint with weights_only=False to handle numpy scalars
        print("Loading checkpoint with weights_only=False...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("Checkpoint loaded successfully")
        
        # Extract network weights
        if isinstance(checkpoint, dict) and "network_weights" in checkpoint:
            state_dict = checkpoint["network_weights"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Load weights
        trainer.network.load_state_dict(state_dict)
        print("Checkpoint weights loaded into trainer network.")

        if hasattr(trainer.network, 'deep_supervision'):
            trainer.network.deep_supervision = False
            print("Disabled deep supervision for inference to fix shape mismatch.")
        
        # Set network to evaluation mode and enable fast inference
        trainer.network.eval()
        if hasattr(trainer.network, 'enable_fast_inference'):
            trainer.network.enable_fast_inference()
            print("Enabled fast inference mode")
        
        # Set up nnUNetPredictor with the loaded model
        predictor.network = trainer.network
        predictor.plans_manager = plans_manager
        predictor.configuration_manager = configuration_manager
        predictor.dataset_json = dataset_json
        predictor.label_manager = label_manager
        predictor.trainer_name = "MultitasknnUNetTrainer"
        predictor.allowed_mirroring_axes = None  # Default
        
        # CRITICAL FIX: Initialize the list_of_parameters property which is needed for inference
        predictor.list_of_parameters = [state_dict]
        print(f"Target spacing for resampling: {configuration_manager.spacing}")

    except Exception as e:
        print(f"Error during initialization: {e}")
        traceback.print_exc()
        return False

    # Process test cases
    print("\nProcessing test cases individually with manual preprocessing...")
    
    # Create predictions directory
    maybe_mkdir_p(output_folder)
    
    # Initialize array for classification results
    classification_results = []

    # Process files individually to better handle any shape issues
    from tqdm import tqdm
    for case_idx, (input_file, case_id) in enumerate(tqdm(zip(input_files, case_ids), total=len(input_files), desc="Processing Cases")):
        output_file = join(output_folder, f"{case_id}.nii.gz")
        
        try:
            # Create individual lists for this case
            input_files_list = [[input_file]]
            output_files_list = [output_file]
            
            # Run prediction with official nnUNetPredictor
            predictor.predict_from_files(
                input_files_list,
                output_files_list,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )
            
            # Get classification prediction
            predicted_class = 0  # Default
            if hasattr(trainer.network, 'last_classification_output') and trainer.network.last_classification_output is not None:
                cls_output = trainer.network.last_classification_output
                probs = F.softmax(cls_output, dim=1)[0].cpu().numpy()
                predicted_class = int(np.argmax(probs))
            
            # Add to classification results
            classification_results.append((f"{case_id}.nii.gz", predicted_class))
            print(f"Case {case_id}: Predicted class {predicted_class}")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"Error processing case {case_id} with nnUNetPredictor: {e}")
            
            # Fallback to manual processing for problematic cases
            try:
                print(f"Trying manual preprocessing for case {case_id}...")
                
                # Manual preprocessing similar to nnUNetPredictor but with more flexibility
                # Read image
                images, properties = reader_writer.read_images([input_file])
                
                # Manual preprocessing
                from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
                preprocessor = DefaultPreprocessor(verbose=False)
                
                try:
                    data, seg, properties = preprocessor.run_case_npy(
                        images, 
                        None, 
                        properties, 
                        plans_manager, 
                        configuration_manager, 
                        dataset_json
                    )
                except Exception as preprocess_error:
                    print(f"Warning: Standard preprocessing failed: {preprocess_error}")
                    print("Applying Z-score normalization as fallback...")
                    
                    # Apply simple Z-score normalization as fallback
                    data = images
                    for c in range(data.shape[0]):
                        mean = data[c].mean()
                        std = data[c].std()
                        data[c] = (data[c] - mean) / (std + 1e-8)
                
                # Convert to tensor
                data = torch.from_numpy(data).cuda().float()
                
                # Run inference with specific error handling
                with torch.no_grad():
                    # Use a sliding window approach manually if needed
                    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
                    from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
                    
                    # Ensure model is in eval mode
                    trainer.network.eval()
                    
                    # Run sliding window prediction with patch sizes that fit
                    patch_size = configuration_manager.patch_size
                    data_shape = data.shape[1:]
                    
                    # Pad data to ensure it can accommodate the patch size
                    if any(i < j for i, j in zip(data_shape, patch_size)):
                        pad_size = [(max(0, p - d)) for d, p in zip(data_shape, patch_size)]
                        pad_size = [(0, p) for p in pad_size]
                        data = np.pad(data, [(0, 0)] + pad_size, mode='constant')
                    
                    # Get steps for sliding window
                    steps = compute_steps_for_sliding_window(data.shape[1:], patch_size, 0.5)
                    
                    # Run inference
                    prediction = trainer.network(data[None])
                    
                    # Export result
                    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
                    seg_pred = convert_predicted_logits_to_segmentation_with_correct_shape(
                        prediction.cpu(), 
                        plans_manager,
                        configuration_manager, 
                        label_manager,
                        properties, 
                        save_probabilities=False
                    )
                    
                    # Save segmentation
                    reader_writer.write_seg(seg_pred, output_file, properties)
                    
                    # Get classification
                    if hasattr(trainer.network, 'last_classification_output') and trainer.network.last_classification_output is not None:
                        cls_output = trainer.network.last_classification_output
                        probs = F.softmax(cls_output, dim=1)[0].cpu().numpy()
                        predicted_class = int(np.argmax(probs))
                    else:
                        predicted_class = 0
                    
                    classification_results.append((f"{case_id}.nii.gz", predicted_class))
                    print(f"Case {case_id} successfully processed with fallback method. Predicted class: {predicted_class}")
            
            except Exception as fallback_error:
                print(f"Segmentation failed for {case_id}: {fallback_error}")

    
    # Save classification results
    csv_path = join(output_folder, "subtype_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Names', 'Subtype'])
        for name, subtype in classification_results:
            writer.writerow([name, subtype])
    
    print(f"\nResults saved to {output_folder}")
    print(f"Classification results saved to {csv_path}")
    
    # Print timing info
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nInference completed in {total_time:.2f} seconds")
    print(f"Average time per case: {total_time / len(input_files):.2f} seconds")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run improved inference for pancreatic cancer segmentation and classification")
    parser.add_argument("--task_id", type=int, default=900, help="Task ID")
    parser.add_argument("--test_folder", type=str, required=True, help="Folder containing test images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save prediction results")
    parser.add_argument("--fold", type=int, default=0, help="Fold to use")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth", help="Checkpoint filename")
    args = parser.parse_args()
    
    improved_inference(
        task_id=args.task_id, 
        test_folder=args.test_folder, 
        output_folder=args.output_folder, 
        fold=args.fold, 
        checkpoint_name=args.checkpoint
    )
