#!/usr/bin/env python3
"""
Final inference script for pancreatic cancer segmentation and classification
Submission-Ready:
    - Reads the actual quiz test files (quiz_XXX_0000.nii.gz)
    - Produces segmentation masks (quiz_XXX.nii.gz)
    - Produces classification results in subtype_results.csv
    - Disables deep supervision to avoid shape mismatches
    - Implements optional fallback for low-memory cases
"""

import os
import sys
import time
import csv
import glob
import torch
import traceback
import numpy as np
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json, isfile

# 1) Disable TorchDynamo before anything else
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Handle functional calls for classification
import torch.nn.functional as F

def inference(
    task_id=900,
    test_folder="/content/gdrive/MyDrive/PancreasCancerFinal/dataset/test",
    output_folder="./your_name_results",
    fold=0,
    checkpoint_name="checkpoint_best.pth"
):
    """
    Final inference pipeline that:
      - Loads a trained MultitasknnUNetTrainer model
      - Disables deep supervision (fix shape mismatch)
      - Processes each test case to produce:
         (A) quiz_xxx.nii.gz (segmentation file)
         (B) subtype_results.csv (classification)

    Args:
        task_id (int): The task ID used by nnUNetv2 (default: 900)
        test_folder (str): Folder containing final quiz test images
        output_folder (str): Folder where final results are saved
        fold (int): Fold index; usually 0
        checkpoint_name (str): Name of the model checkpoint to load
    """
    start_time = time.time()

    print("\n========== FINAL INFERENCE PIPELINE ==========")
    print(f"Test folder:      {test_folder}")
    print(f"Output folder:    {output_folder}")
    print(f"Fold:             {fold}")
    print(f"Checkpoint:       {checkpoint_name}")
    print("================================================\n")

    # Make sure the output directory exists
    maybe_mkdir_p(output_folder)

    # Ensure current working directory is in Python path
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    # 2) Import the trainer
    try:
        from multitask_trainer import MultitasknnUNetTrainer
        print("Successfully imported MultitasknnUNetTrainer")
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Please ensure 'multitask_trainer.py' is in the same directory or in sys.path.")
        return False

    # 3) Locate the trained model checkpoint
    nnunet_results = os.environ.get("nnUNet_results", "/content/output/nnUNet_results")
    model_folder = join(
        nnunet_results,
        f"Dataset{task_id:03d}_PancreasCancer",
        "MultitasknnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
    )
    fold_dir = join(model_folder, f"fold_{fold}")
    checkpoint_path = join(fold_dir, checkpoint_name)

    # 4) Find quiz files in test folder
    input_files, case_ids = [], []
    print(f"Searching for quiz files in: {test_folder}")
    if os.path.isdir(test_folder):
        # If subfolders exist, check them
        subtype_dirs = [
            d for d in os.listdir(test_folder)
            if os.path.isdir(join(test_folder, d)) and "subtype" in d
        ]
        if subtype_dirs:
            for sd in subtype_dirs:
                subtype_path = join(test_folder, sd)
                for f in glob.glob(join(subtype_path, "*_0000.nii.gz")):
                    cid = os.path.basename(f).replace("_0000.nii.gz", "")
                    input_files.append(f)
                    case_ids.append(cid)
        else:
            # Directly in the test folder
            for f in glob.glob(join(test_folder, "*_0000.nii.gz")):
                cid = os.path.basename(f).replace("_0000.nii.gz", "")
                input_files.append(f)
                case_ids.append(cid)

    if len(input_files) == 0:
        print(f"No quiz NIfTI files found in {test_folder}. Nothing to do.")
        return False
    else:
        print(f"Found {len(input_files)} quiz files for final inference.")

    # 5) Load nnU-Net Plans & Dataset config
    plans_file = join(model_folder, "plans.json")
    dataset_json_file = join(model_folder, "dataset.json")

    # Fallback if not found
    if not isfile(plans_file) or not isfile(dataset_json_file):
        print("Error: Could not locate 'plans.json' or 'dataset.json' in the model folder.")
        print(f"Plans:   {plans_file}")
        print(f"Dataset: {dataset_json_file}")
        return False

    print(f"Loading plans from:   {plans_file}")
    print(f"Loading dataset from: {dataset_json_file}")
    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)

    try:
        # 6) Create PlansManager, etc.
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
        from nnunetv2.utilities.label_handling.label_handling import LabelManager
        from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration("3d_fullres")
        label_manager = plans_manager.get_label_manager(dataset_json)

        # 7) Create an image reader/writer for the test scans
        reader_writer_class = determine_reader_writer_from_dataset_json(dataset_json)
        reader_writer = reader_writer_class()

        # 8) Instantiate nnUNetPredictor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inference device: {device}")

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

        # 9) Initialize the trainer and load the checkpoint
        print("Loading the MultitasknnUNetTrainer for final inference...")
        trainer = MultitasknnUNetTrainer(
            plans=plans,
            configuration="3d_fullres",
            fold=fold,
            dataset_json=dataset_json,
            device=device
        )
        trainer.initialize()

        # Load the checkpoint
        print("Loading checkpoint with weights_only=False...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("Checkpoint loaded successfully.")

        # Extract network weights
        if isinstance(ckpt, dict):
            if "network_weights" in ckpt:
                state_dict = ckpt["network_weights"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Load them into the model
        trainer.network.load_state_dict(state_dict)
        print("Checkpoint weights loaded into trainer network.")

        # Disable deep supervision (fix shape mismatch)
        if hasattr(trainer.network, "deep_supervision"):
            trainer.network.deep_supervision = False
            print("Disabled deep supervision for inference.")

        # Put the model in eval mode, enable speed-ups
        trainer.network.eval()
        if hasattr(trainer.network, "enable_fast_inference"):
            trainer.network.enable_fast_inference()
            print("Enabled fast inference mode.")

        # Assign model & config to predictor
        predictor.network = trainer.network
        predictor.plans_manager = plans_manager
        predictor.configuration_manager = configuration_manager
        predictor.dataset_json = dataset_json
        predictor.label_manager = label_manager
        predictor.trainer_name = "MultitasknnUNetTrainer"
        predictor.allowed_mirroring_axes = None
        predictor.list_of_parameters = [state_dict]

    except Exception as e:
        print(f"Error during initialization: {e}")
        traceback.print_exc()
        return False

    # 10) Process each quiz file, generating segmentation + classification
    classification_results = []
    maybe_mkdir_p(output_folder)

    from tqdm import tqdm
    print("\nRunning inference on quiz cases...")
    for idx, (file_in, case_id) in enumerate(tqdm(zip(input_files, case_ids),
                                                  total=len(input_files),
                                                  desc="Final Inference")):
        file_out = join(output_folder, f"{case_id}.nii.gz")

        try:

            input_files_list = [[file_in]]
            output_files_list = [file_out]
            
            # Official predictor usage
            predictor.predict_from_files(
                input_files_list,               # 2D list
                output_files_list,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )

            # Retrieve classification
            predicted_class = 0
            if (hasattr(trainer.network, "last_classification_output")
                    and trainer.network.last_classification_output is not None):
                with torch.no_grad():
                    cls_out = trainer.network.last_classification_output
                    probs = F.softmax(cls_out, dim=1)[0].cpu().numpy()
                    predicted_class = int(np.argmax(probs))

            classification_results.append((f"{case_id}.nii.gz", predicted_class))
            print(f"Case {case_id}: predicted class {predicted_class}")
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"[ERROR] Inference failure for {case_id} with nnUNetPredictor: {e}")

    # 11) Save classification results to CSV
    csv_path = join(output_folder, "subtype_results.csv")
    with open(csv_path, "w", newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(["Names", "Subtype"])
        for name, subtype in classification_results:
            writer.writerow([name, subtype])

    print(f"\nInference results saved to: {output_folder}")
    print(f"Classification written to:  {csv_path}")

    # 12) Timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nInference completed in {total_time:.2f}s")
    print(f"Average time per case: {total_time / len(input_files):.2f}s")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run final inference for the quiz test files.")
    parser.add_argument("--task_id", type=int, default=900, help="nnUNet task ID (default: 900)")
    parser.add_argument("--test_folder", type=str, required=True, help="Folder containing quiz test images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save quiz results")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to load (default: 0)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth", help="Model checkpoint name")
    args = parser.parse_args()

    inference(
        task_id=args.task_id,
        test_folder=args.test_folder,
        output_folder=args.output_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint
    )
