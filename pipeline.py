#!/usr/bin/env python3
"""
Modified pipeline script for pancreatic cancer segmentation and classification
with custom preprocessing to force using the original validation set
"""

import os
import sys
import glob
import argparse
import subprocess
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
import importlib.util
from batchgenerators.utilities.file_and_folder_operations import join, save_json, load_json

def run_command(cmd, verbose=True):
    """Run a command and print output"""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )

    # Stream the output
    output = ""
    for line in process.stdout:
        if verbose:
            print(line.strip())
        output += line

    process.wait()
    return process.returncode, output


def setup_environment(base_dir):
    """Set up environment variables and create necessary directories"""
    dataset_dir = os.path.join(base_dir, "dataset")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    os.environ["nnUNet_raw"] = os.path.join(output_dir, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(output_dir, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(output_dir, "nnUNet_results")

    os.makedirs(os.environ["nnUNet_raw"], exist_ok=True)
    os.makedirs(os.environ["nnUNet_preprocessed"], exist_ok=True)
    os.makedirs(os.environ["nnUNet_results"], exist_ok=True)

    print(f"Environment set up successfully:")
    print(f"  Dataset directory: {dataset_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  nnUNet_raw: {os.environ['nnUNet_raw']}")
    print(f"  nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"  nnUNet_results: {os.environ['nnUNet_results']}")

    return dataset_dir, output_dir


def fix_label_files(input_dir):
    """
    Fix all label files in the directory by rounding to integers
    """
    # Find all label files (those without _0000 in the name)
    label_files = glob.glob(os.path.join(input_dir, "**", "*.nii.gz"), recursive=True)
    label_files = [f for f in label_files if "_0000" not in f]

    print(f"Found {len(label_files)} label files to process")

    for file_path in label_files:
        try:
            # Load image
            img = nib.load(file_path)
            data = img.get_fdata()

            # Check if it needs fixing (has non-integer values)
            unique_vals = np.unique(data)
            non_int_vals = [v for v in unique_vals if not np.isclose(v, np.round(v))]

            if len(non_int_vals) > 0:
                print(f"Fixing {file_path}, non-integer values: {non_int_vals}")

                # Round to nearest integer and convert to uint8
                data_fixed = np.round(data).astype(np.uint8)

                # Create new image and save
                new_img = nib.Nifti1Image(data_fixed, img.affine, img.header)
                nib.save(new_img, file_path)
                print(f"  - Fixed. New values: {np.unique(data_fixed)}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def prepare_dataset(dataset_dir, output_dir, task_id, task_name):
    """Prepares the pancreatic cancer dataset for nnUNetv2 by organizing into the required structure.
    Maintains separation between training and validation data."""
    print("\n=== Preparing dataset ===")
    cmd = [
        sys.executable,
        "dataset_preparation.py",
        "-i", dataset_dir,
        "-o", os.environ["nnUNet_raw"],
        "--task_id", str(task_id),
        "--task_name", task_name
    ]
    ret_code, output = run_command(cmd)

    # Fix labels in the prepared dataset
    if ret_code == 0:
        dataset_path = os.path.join(os.environ["nnUNet_raw"], f"Dataset{task_id:03d}_{task_name}")
        print(f"Fixing labels in {dataset_path}")
        fix_label_files(dataset_path)
        return True
    else:
        print("Dataset preparation failed")
        return False


def collect_validation_identifiers(dataset_dir):
    """Collect the original validation identifiers from the dataset structure"""
    validation_ids = []
    for subtype in ["subtype0", "subtype1", "subtype2"]:
        subtype_dir = os.path.join(dataset_dir, "validation", subtype)
        if os.path.exists(subtype_dir):
            for file in glob.glob(os.path.join(subtype_dir, "*_0000.nii.gz")):
                case_id = os.path.basename(file)[:-len("_0000.nii.gz")]
                validation_ids.append(case_id)

    return validation_ids


def collect_training_identifiers(dataset_dir):
    """Collect the original training identifiers from the dataset structure"""
    training_ids = []
    for subtype in ["subtype0", "subtype1", "subtype2"]:
        subtype_dir = os.path.join(dataset_dir, "train", subtype)
        if os.path.exists(subtype_dir):
            for file in glob.glob(os.path.join(subtype_dir, "*_0000.nii.gz")):
                case_id = os.path.basename(file)[:-len("_0000.nii.gz")]
                training_ids.append(case_id)

    return training_ids


def get_preprocessed_cases(task_id, task_name):
    """Get all available preprocessed cases"""
    preprocessed_dir = os.path.join(os.environ["nnUNet_preprocessed"], f"Dataset{task_id:03d}_{task_name}")
    available_cases = set()

    # Find all preprocessed directories
    preprocessed_folders = []
    for pattern in ["nnUNetPlans*", "nnUNetResEncUNetMPlans*"]:
        folder_matches = glob.glob(os.path.join(preprocessed_dir, pattern))
        for folder in folder_matches:
            if os.path.isdir(folder):
                preprocessed_folders.append(folder)

    # Get all case IDs from each folder
    for folder in preprocessed_folders:
        for file in glob.glob(os.path.join(folder, "*.b2nd")):
            if "_seg" not in file:  # Skip segmentation files
                case_id = os.path.basename(file).split('.')[0]
                available_cases.add(case_id)

    return available_cases


def create_custom_splits(dataset_dir, output_dir, task_id, task_name):
    """Create a custom split file using validation identifiers saved during dataset preparation"""
    print("\n=== Creating custom split file using original validation set ===")

    # Path to saved validation identifiers
    dataset_path = os.path.join(os.environ["nnUNet_raw"], f"Dataset{task_id:03d}_{task_name}")
    val_data_file = os.path.join(dataset_path, "validation_identifiers.json")

    if not os.path.exists(val_data_file):
        print(f"Validation identifiers file not found: {val_data_file}")
        print("Falling back to folder structure to identify validation cases")

        # Get original validation and training identifiers from folder structure
        validation_ids = []
        for subtype in ["subtype0", "subtype1", "subtype2"]:
            subtype_dir = os.path.join(dataset_dir, "validation", subtype)
            if os.path.exists(subtype_dir):
                for file in glob.glob(os.path.join(subtype_dir, "*_0000.nii.gz")):
                    case_id = os.path.basename(file)[:-len("_0000.nii.gz")]
                    validation_ids.append(case_id)

        training_ids = []
        for subtype in ["subtype0", "subtype1", "subtype2"]:
            subtype_dir = os.path.join(dataset_dir, "train", subtype)
            if os.path.exists(subtype_dir):
                for file in glob.glob(os.path.join(subtype_dir, "*_0000.nii.gz")):
                    case_id = os.path.basename(file)[:-len("_0000.nii.gz")]
                    training_ids.append(case_id)
    else:
        # Load validation and training identifiers from saved file
        val_data = load_json(val_data_file)
        validation_ids = val_data["validation_identifiers"]
        training_ids = val_data["training_identifiers"]

    print(f"Original validation cases: {len(validation_ids)}")
    print(f"Original training cases: {len(training_ids)}")

    # Get available preprocessed cases to make sure we only include cases that were preprocessed
    available_cases = get_preprocessed_cases(task_id, task_name)
    print(f"Found {len(available_cases)} available preprocessed cases")

    # Filter to only include preprocessed cases
    valid_val_ids = [id for id in validation_ids if id in available_cases]
    valid_train_ids = [id for id in training_ids if id in available_cases]

    print(f"Preprocessed validation cases: {len(valid_val_ids)}")
    print(f"Preprocessed training cases: {len(valid_train_ids)}")

    # Check if we have validation cases
    if len(valid_val_ids) == 0:
        print("WARNING: No validation cases found in preprocessed data!")
        print("This indicates that validation cases weren't properly preprocessed.")
        print("Falling back to using 20% of training cases for validation.")

        # Use 20% of training cases for validation as fallback
        import random
        random.seed(42)  # For reproducibility
        all_cases = valid_train_ids.copy()
        random.shuffle(all_cases)
        val_size = max(1, int(0.2 * len(all_cases)))
        valid_val_ids = all_cases[:val_size]
        valid_train_ids = all_cases[val_size:]

    # Create the splits file
    preprocessed_dir = os.path.join(os.environ["nnUNet_preprocessed"], f"Dataset{task_id:03d}_{task_name}")
    splits_file = os.path.join(preprocessed_dir, "splits_final.json")

    # Create a single split for our custom train/val
    splits = [{"train": valid_train_ids, "val": valid_val_ids}]

    # Save the splits file
    print(f"Saving custom split to {splits_file}")
    print(f"Training cases: {len(valid_train_ids)}")
    print(f"Validation cases: {len(valid_val_ids)}")

    # Make sure the directory exists
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Delete existing splits file if it exists (to avoid appending)
    if os.path.exists(splits_file):
        os.remove(splits_file)

    save_json(splits, splits_file)

    # Verify the saved file has only one split
    loaded_splits = load_json(splits_file)
    print(f"Verified: Created split file contains {len(loaded_splits)} split(s)")

    return splits_file


def plan_and_preprocess(task_id, planner="nnUNetPlannerResEncM"):
    """Run experiment planning and preprocessing with numeric dataset ID"""
    print("\n=== Planning and preprocessing ===")

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(task_id),  # Use numeric ID only
        "-pl", planner,
        "--verify_dataset_integrity"
    ]

    ret_code, output = run_command(cmd)
    if ret_code != 0:
        print("Preprocessing failed. Checking for label issues...")
        # Check if there are still label issues
        if "Unexpected labels found" in output:
            print("There are still label issues. Trying to fix them...")
            dataset_path = os.path.join(os.environ["nnUNet_raw"], f"Dataset{task_id:03d}_PancreasCancer")
            fix_label_files(dataset_path)

            # Try preprocessing again
            print("Running preprocessing again after fixing labels...")
            ret_code, output = run_command(cmd)
            if ret_code != 0:
                print("Preprocessing still failed after fixing labels.")
                return False

    return ret_code == 0


def preprocess_validation_data(task_id, task_name, configuration="3d_fullres"):
    """Preprocess validation data specifically"""
    print("\n=== Preprocessing validation data ===")

    # First, check if imagesVal folder exists
    val_folder = os.path.join(os.environ["nnUNet_raw"], f"Dataset{task_id:03d}_{task_name}", "imagesVal")
    if not os.path.exists(val_folder) or len(os.listdir(val_folder)) == 0:
        print(f"Validation folder {val_folder} doesn't exist or is empty. Cannot preprocess validation data.")
        return False

    # Run preprocessing for validation data
    cmd = [
        "nnUNetv2_preprocess",
        "-d", str(task_id),
        "-c", configuration,
        "--validate_only"
    ]

    ret_code, output = run_command(cmd)
    return ret_code == 0


def run_python_script_for_training(task_id, fold, configuration="3d_fullres", trainer="MultitasknnUNetTrainer", plans="nnUNetResEncUNetMPlans"):
    """Use a direct Python script approach for training to ensure the trainer is properly loaded"""
    print(f"\n=== Training {configuration} fold {fold} using direct Python script ===")

    # Format the dataset name directly here, without using an f-string in the script
    dataset_name = f"Dataset{task_id:03d}_PancreasCancer"

    # Create a temporary Python script that imports and uses the trainer directly
    script_content = f"""
import sys
import os

# Add current directory to path to find our custom trainer
sys.path.append(os.getcwd())
from multitask_trainer import MultitasknnUNetTrainer

# Import necessary nnUNetv2 components
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

# Setup parameters
dataset_id = {task_id}
configuration = "{configuration}"
fold = {fold}
plans_identifier = "{plans}"

# Load dataset.json - use the pre-formatted dataset name
dataset_name = "{dataset_name}"
dataset_json = load_json(join(nnUNet_raw, dataset_name, "dataset.json"))

# Load plans
plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
plans = load_json(plans_file)

# Create trainer instance directly with the correct parameters
trainer = MultitasknnUNetTrainer(
    plans=plans,
    configuration=configuration,
    fold=fold,
    dataset_json=dataset_json,
)

trainer.num_epochs = 500  # Set a reasonable limit (original 300 may be too much)
trainer.early_stopping_patience = 100  # Stop after 20 epochs without improvement (original 30)

# Run training
trainer.run_training()

print("Training completed successfully")
"""

    temp_script_path = f"train_fold_{fold}.py"
    with open(temp_script_path, 'w') as f:
        f.write(script_content)

    # Run the script
    cmd = [sys.executable, temp_script_path]
    return run_command(cmd)


def train(task_id, fold, configuration="3d_fullres", trainer="MultitasknnUNetTrainer", plans="nnUNetResEncUNetMPlans"):
    """Try training with nnUNetv2_train first, fall back to direct script if it fails"""
    print(f"\n=== Training {configuration} fold {fold} ===")

    # Skip the nnUNetv2_train command and directly use our script approach
    # This ensures we use our custom MultitasknnUNetTrainer implementation
    ret_code, output = run_python_script_for_training(task_id, fold, configuration, trainer, plans)

    # Copy dataset.json to results folder for inference
    if ret_code == 0:
        copy_dataset_json_to_results(task_id, configuration, trainer, plans)

    return ret_code == 0


def copy_dataset_json_to_results(task_id, configuration, trainer, plans):
    """Copy dataset.json to results folder for inference"""
    print("\n=== Copying dataset.json to results folder ===")
    source_path = os.path.join(os.environ["nnUNet_raw"], f"Dataset{task_id:03d}_PancreasCancer", "dataset.json")
    target_dir = os.path.join(os.environ["nnUNet_results"],
                         f"Dataset{task_id:03d}_PancreasCancer",
                         f"{trainer}__{plans}__{configuration}")
    target_path = os.path.join(target_dir, "dataset.json")

    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
        print(f"Copied dataset.json to {target_path}")
        return True
    else:
        print(f"Source dataset.json not found at {source_path}")
        return False




def run_optimized_inference(
    task_id=900,
    test_folder="/content/gdrive/MyDrive/PancreasCancerFinal/dataset/test",
    output_folder="/content/gdrive/MyDrive/PancreasCancerFinal/output/inference_results",
    fold=0
):
    """
    Fixed whole-image inference with proper padding to handle
    UNet architecture requirements for divisibility
    """
    import os
    import time
    import torch
    import torch.nn.functional as F
    import numpy as np
    import SimpleITK as sitk
    from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
    from tqdm import tqdm
    import csv
    import matplotlib.pyplot as plt
    import shutil

    torch._dynamo.disable()

    print("Starting fixed whole-image inference for A100 GPU...")
    start_time = time.time()

    # Create output directories
    maybe_mkdir_p(output_folder)
    local_output = "/tmp/inference_results"
    maybe_mkdir_p(local_output)

    # Get device - force CUDA for A100
    device = torch.device("cuda")
    print(f"Using device: {device}")



    # Setup paths
    nnunet_results = os.environ.get("nnUNet_results", "/content/gdrive/MyDrive/PancreasCancerFinal/output/nnUNet_results")
    model_folder = join(nnunet_results, f"Dataset{task_id:03d}_PancreasCancer/MultitasknnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
    fold_dir = join(model_folder, f"fold_{fold}")

    # Import necessary modules
    import sys
    sys.path.append(os.getcwd())
    from multitask_trainer import MultitasknnUNetTrainer
    from batchgenerators.utilities.file_and_folder_operations import load_json

    # Load configuration files
    print("Loading configuration files...")
    dataset_json_file = join(model_folder, "dataset.json")
    plans_file = join(model_folder, "plans.json")

    try:
        dataset_json = load_json(dataset_json_file)
        plans = load_json(plans_file)
    except:
        # Fallback paths
        nnunet_raw = os.environ.get("nnUNet_raw", "/content/gdrive/MyDrive/PancreasCancerFinal/output/nnUNet_raw")
        nnunet_preprocessed = os.environ.get("nnUNet_preprocessed", "/content/gdrive/MyDrive/PancreasCancerFinal/output/nnUNet_preprocessed")

        dataset_name = f"Dataset{task_id:03d}_PancreasCancer"
        dataset_json_file = join(nnunet_raw, dataset_name, "dataset.json")
        plans_file = join(nnunet_preprocessed, dataset_name, "nnUNetResEncUNetMPlans.json")

        print(f"Using fallback paths: {dataset_json_file}, {plans_file}")
        dataset_json = load_json(dataset_json_file)
        plans = load_json(plans_file)

    # Initialize trainer
    print("Initializing trainer...")
    trainer = MultitasknnUNetTrainer(
        plans=plans,
        configuration="3d_fullres",
        fold=fold,
        dataset_json=dataset_json,
        device=device
    )

    # Initialize network
    print("Initializing network...")
    trainer.initialize()

    # Load checkpoint with error handling
    checkpoint_file = join(fold_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_file):
        checkpoint_file = join(fold_dir, "checkpoint_final.pth")

    print(f"Loading checkpoint: {checkpoint_file}")

    try:
        # FIX: Add all numpy-related types to safe globals
        import torch.serialization
        import numpy as np
        from numpy import dtype, ndarray, float32, float64, int64, int32, uint8, bool_

        # Add numpy types to safe globals
        torch.serialization.add_safe_globals([
            dtype, ndarray, float32, float64, int32, int64, uint8, bool_,
            np.dtype, np.ndarray, np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_
        ])

        # Try with weights_only=False explicitly
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "network_weights" in checkpoint:
            trainer.network.load_state_dict(checkpoint["network_weights"])
        else:
            trainer.network.load_state_dict(checkpoint)

        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise RuntimeError("Failed to load checkpoint - cannot proceed")

    # Set network to evaluation mode
    trainer.network.eval()

    # Enable fast inference mode if available
    if hasattr(trainer.network, 'enable_fast_inference'):
        trainer.network.enable_fast_inference()
        print("Enabled fast inference mode")

    # Get expected input size from network - IMPORTANT FOR PROPER PADDING
    if hasattr(trainer.configuration_manager, 'patch_size'):
        patch_size = trainer.configuration_manager.patch_size
    else:
        # Default patch size if not available
        patch_size = np.array([64, 64, 64])

    # Calculate required divisibility for the network
    # This depends on the number of pooling operations
    if hasattr(trainer.configuration_manager, 'pool_op_kernel_sizes'):
        pool_op_kernel_sizes = trainer.configuration_manager.pool_op_kernel_sizes
        num_pool_per_axis = [len(pool_op_kernel_sizes) for _ in range(3)]
    else:
        # Default to 5 downsampling operations if not available
        num_pool_per_axis = [5, 5, 5]

    # Calculate divisibility factor for each dimension
    divisibility_factor = [2 ** num_pool for num_pool in num_pool_per_axis]
    print(f"Network requires input dimensions divisible by: {divisibility_factor}")

    # Find test files
    if os.path.isdir(test_folder):
        test_files = [join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('_0000.nii.gz')]
        if not test_files:
            test_files = [join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.nii.gz')]
    else:
        test_files = [test_folder]

    print(f"Found {len(test_files)} test files")

    # Process test files
    classification_results = []
    class_distributions = [0, 0, 0]  # Count of predictions for each class

    for test_file in tqdm(test_files, desc="Processing test files"):
        # Get case ID
        base_name = os.path.basename(test_file)
        if "_0000.nii.gz" in base_name:
            case_id = base_name.replace("_0000.nii.gz", "")
        else:
            case_id = base_name.replace(".nii.gz", "")

        # Load image
        img = sitk.ReadImage(test_file)
        spacing = img.GetSpacing()
        direction = img.GetDirection()
        origin = img.GetOrigin()

        # Convert to numpy array
        image_data = sitk.GetArrayFromImage(img)
        original_shape = image_data.shape

        # Normalize
        image_data = image_data.astype(np.float32)
        mean = np.mean(image_data)
        std = np.std(image_data)
        if std > 0:
            image_data = (image_data - mean) / std

        # Calculate padding to make dimensions divisible
        padded_shape = []
        pad_amounts = []

        for i, dim_size in enumerate(image_data.shape):
            # Calculate needed size (next multiple of divisibility factor)
            needed_size = int(np.ceil(dim_size / divisibility_factor[i]) * divisibility_factor[i])
            padded_shape.append(needed_size)

            # Calculate padding (before and after)
            pad_before = (needed_size - dim_size) // 2
            pad_after = needed_size - dim_size - pad_before
            pad_amounts.append((pad_before, pad_after))

        print(f"Case {case_id}: Original shape {original_shape}, padded to {padded_shape}")

        # Pad the image
        padded_image = np.pad(image_data,
                             ((pad_amounts[0][0], pad_amounts[0][1]),
                              (pad_amounts[1][0], pad_amounts[1][1]),
                              (pad_amounts[2][0], pad_amounts[2][1])),
                             mode='constant', constant_values=0)

        # Convert to tensor and add batch and channel dimensions
        image_tensor = torch.from_numpy(padded_image).unsqueeze(0).unsqueeze(0).to(device)

        # Process the whole image at once with mixed precision
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                try:
                    # Forward pass
                    output = trainer.network(image_tensor)

                    # Get segmentation prediction
                    if isinstance(output, list):
                        seg_output = output[0]
                    else:
                        seg_output = output

                    # Get classification prediction
                    if hasattr(trainer.network, 'last_classification_output'):
                        cls_output = trainer.network.last_classification_output
                        probs = F.softmax(cls_output, dim=1)[0].cpu().numpy()
                        predicted_class = np.argmax(probs)
                        confidence = probs[predicted_class]
                        print(f"Case {case_id}: Classification probabilities = {probs}, predicted class = {predicted_class}, confidence = {confidence:.4f}")
                    else:
                        # Try direct classification head access
                        cls_output = trainer.network.classification_head(trainer.network.bottleneck_features)
                        probs = F.softmax(cls_output, dim=1)[0].cpu().numpy()
                        predicted_class = np.argmax(probs)
                        confidence = probs[predicted_class]

                    # Unpad the segmentation output
                    seg_output_numpy = torch.argmax(seg_output, dim=1)[0].cpu().numpy()

                    # Unpad to original shape
                    unpadded_seg = seg_output_numpy[
                        pad_amounts[0][0]:pad_amounts[0][0]+original_shape[0],
                        pad_amounts[1][0]:pad_amounts[1][0]+original_shape[1],
                        pad_amounts[2][0]:pad_amounts[2][0]+original_shape[2]
                    ]

                except Exception as e:
                    print(f"Error processing case {case_id}: {e}")
                    continue

        # Update class distribution
        class_distributions[predicted_class] += 1

        # Convert segmentation to SimpleITK image
        seg_itk = sitk.GetImageFromArray(unpadded_seg)
        seg_itk.SetSpacing(spacing)
        seg_itk.SetDirection(direction)
        seg_itk.SetOrigin(origin)

        # Save to output directory
        local_output_file = join(local_output, f"{case_id}.nii.gz")
        sitk.WriteImage(seg_itk, local_output_file)

        # Add to classification results
        classification_results.append((f"{case_id}.nii.gz", predicted_class))

    # Save classification results
    csv_path = join(local_output, "subtype_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Names', 'Subtype'])
        for name, subtype in classification_results:
            writer.writerow([name, subtype])

    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(['Subtype 0', 'Subtype 1', 'Subtype 2'], class_distributions)
    plt.title('Distribution of Predicted Classes')
    plt.savefig(join(local_output, 'class_distribution.png'))
    print(f"Final class distribution: {class_distributions}")

    # Copy files to final destination
    print(f"Copying results from {local_output} to {output_folder}")
    for file_name in os.listdir(local_output):
        src_file = join(local_output, file_name)
        dst_file = join(output_folder, file_name)
        shutil.copy2(src_file, dst_file)

    # Report timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Inference completed in {total_time:.2f} seconds")
    print(f"Average time per case: {total_time/len(test_files):.2f} seconds")
    print(f"Results saved to {output_folder}")

    return True

def main():
    parser = argparse.ArgumentParser(description="Run the complete pipeline for pancreatic cancer segmentation and classification")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing dataset and for outputs")
    parser.add_argument("--task_id", type=int, default=900, help="Task ID (default: 900)")
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare the dataset without training")
    parser.add_argument("--train_only", action="store_true", help="Only train without inference")
    parser.add_argument("--inference_only", action="store_true", help="Only run inference without training")
    parser.add_argument("--fold", type=int, default=0, help="Use fold 0 for our custom split")
    parser.add_argument("--configuration", type=str, default="3d_fullres",
                        help="Configuration to use (default: 3d_fullres)")
    parser.add_argument("--plans", type=str, default="nnUNetResEncUNetMPlans",
                        help="Plans to use (default: nnUNetResEncUNetMPlans)")
    parser.add_argument("--test_input", type=str, default=None,
                        help="Test input folder for inference (default: dataset/test)")
    parser.add_argument("--test_output", type=str, default=None,
                        help="Test output folder for inference results")

    args = parser.parse_args()

    # Setup environment
    dataset_dir, output_dir = setup_environment(args.base_dir)

    # Determine which steps to run
    run_preparation = not (args.train_only or args.inference_only)
    run_training = not (args.prepare_only or args.inference_only)
    run_inference = not (args.prepare_only or args.train_only)

    # Track success of each step
    preprocessing_success = True
    train_success = True

    # Run the pipeline
    if run_preparation:
        # Prepare dataset and fix labels
        prepare_success = prepare_dataset(dataset_dir, output_dir, args.task_id, "PancreasCancer")
        if not prepare_success:
            print("Dataset preparation failed, stopping pipeline.")
            return

        # Run planning and preprocessing for training data
        preprocessing_success = plan_and_preprocess(args.task_id, "nnUNetPlannerResEncM")
        if not preprocessing_success:
            print("Preprocessing failed, stopping pipeline.")
            return

        # Also preprocess validation data explicitly
        validation_preprocessing_success = preprocess_validation_data(args.task_id, "PancreasCancer", args.configuration)
        if not validation_preprocessing_success:
            print("Warning: Validation data preprocessing failed. Will use training data for validation.")

    # Create the custom split using both original training and validation data
    create_custom_splits(dataset_dir, output_dir, args.task_id, "PancreasCancer")

    if run_training and preprocessing_success:
        # We only use fold 0 which contains our custom split
        train_success = train(args.task_id, 0, args.configuration, "MultitasknnUNetTrainer", args.plans)

        # Only find best configuration if training was successful
        if train_success:
            copy_dataset_json_to_results(args.task_id, args.configuration, "MultitasknnUNetTrainer", args.plans)
        else:
            print("Training failed, skipping best configuration finding and inference.")

    if run_inference and train_success:
        # Determine input and output folders for inference
        test_input = args.test_input if args.test_input else os.path.join(dataset_dir, "test")

        if not os.path.exists(test_input):
            print(f"Test input folder {test_input} does not exist, skipping inference")
        else:
            test_output = args.test_output if args.test_output else os.path.join(output_dir, "inference_results")
            os.makedirs(test_output, exist_ok=True)

            # Run inference (use only fold 0 which has our custom train/validation split)

            inference_success = run_optimized_inference(
                  task_id=900,
                  test_folder="/content/gdrive/MyDrive/PancreasCancerFinal/dataset/test",
                  output_folder="/content/gdrive/MyDrive/PancreasCancerFinal/output/inference_results",
                  fold=0
            )

            if inference_success:
                print(f"Inference completed successfully. Results saved to {test_output}")
            else:
                print("Inference failed.")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()