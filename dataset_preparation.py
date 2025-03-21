#!/usr/bin/env python3
"""
MODIFIED dataset preparation script that includes validation cases in training folder
but keeps track of them separately for proper split creation
"""

import os
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *


def copy_with_metadata_conversion(src_file: str, dst_file: str):
    """
    Copy a NIfTI file with metadata conversion if necessary
    """
    if src_file.endswith('.nii.gz') and dst_file.endswith('.nii.gz'):
        # Load using nibabel to properly handle metadata
        img = nib.load(src_file)

        # Check if this is a label file (without _0000 in the name)
        if "_0000" not in src_file and src_file.endswith('.nii.gz'):
            # Fix labels by rounding to nearest integer
            data = img.get_fdata()
            data = np.round(data).astype(np.uint8)
            # Create a new image with corrected data but same metadata
            new_img = nib.Nifti1Image(data, img.affine, img.header)
            nib.save(new_img, dst_file)
        else:
            # For non-label files, just copy as is
            nib.save(img, dst_file)
    else:
        # Simple copy for other file types
        shutil.copy2(src_file, dst_file)


def prepare_task(input_folder: str, output_folder: str, task_id: int, task_name: str):
    """
    Prepares the pancreatic cancer dataset for nnUNetv2 by organizing into the required structure.
    MODIFIED: Places both training and validation data in the training folder,
    but tracks validation cases for later use in creating custom splits.

    Args:
        input_folder: Path to the original dataset with train/validation/test folders
        output_folder: Path where the nnUNetv2 formatted dataset will be created
        task_id: Task ID (e.g., 900)
        task_name: Task name (e.g., PancreasCancer)
    """
    # Create folders for the nnUNetv2 dataset
    target_base = join(output_folder, f"Dataset{task_id:03d}_{task_name}")
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # Process by dataset split: train, validation, test
    train_identifiers = []
    val_identifiers = []  # We'll track these separately
    test_identifiers = []
    subtypes = ["subtype0", "subtype1", "subtype2"]
    file_ending = ".nii.gz"  # Assuming NIfTI format

    # Process training files
    print("Processing training files...")
    for subtype_idx, subtype in enumerate(subtypes):
        subtype_dir = join(input_folder, "train", subtype)
        if not os.path.exists(subtype_dir):
            print(f"Directory not found: {subtype_dir}, skipping...")
            continue

        # Find all image files in this subtype
        image_files = [f for f in os.listdir(subtype_dir) if f.endswith("_0000.nii.gz")]
        for image_file in image_files:
            # Get case ID from image filename (remove _0000.nii.gz suffix)
            case_id = image_file[:-len("_0000.nii.gz")]

            # Source paths
            image_source = join(subtype_dir, image_file)
            seg_source = join(subtype_dir, f"{case_id}.nii.gz")

            # Only include if segmentation exists
            if not os.path.exists(seg_source):
                print(f"Segmentation missing for {case_id}, skipping...")
                continue

            # Target paths - keep original names to preserve subtype info in filename
            image_target = join(target_imagesTr, image_file)
            seg_target = join(target_labelsTr, f"{case_id}.nii.gz")

            # Copy files
            copy_with_metadata_conversion(image_source, image_target)
            copy_with_metadata_conversion(seg_source, seg_target)

            # Add to training identifiers
            train_identifiers.append(case_id)

    # Process validation files - MODIFIED to add to training folder
    print("Processing validation files - adding to training folder...")
    for subtype_idx, subtype in enumerate(subtypes):
        subtype_dir = join(input_folder, "validation", subtype)
        if not os.path.exists(subtype_dir):
            print(f"Directory not found: {subtype_dir}, skipping...")
            continue

        # Find all image files in this subtype
        image_files = [f for f in os.listdir(subtype_dir) if f.endswith("_0000.nii.gz")]
        for image_file in image_files:
            # Get case ID from image filename (remove _0000.nii.gz suffix)
            case_id = image_file[:-len("_0000.nii.gz")]

            # Source paths
            image_source = join(subtype_dir, image_file)
            seg_source = join(subtype_dir, f"{case_id}.nii.gz")

            # Only include if segmentation exists
            if not os.path.exists(seg_source):
                print(f"Segmentation missing for {case_id}, skipping...")
                continue

            # Target paths - MODIFIED to put in training folder
            image_target = join(target_imagesTr, image_file)
            seg_target = join(target_labelsTr, f"{case_id}.nii.gz")

            # Copy files
            copy_with_metadata_conversion(image_source, image_target)
            copy_with_metadata_conversion(seg_source, seg_target)

            # Add to validation identifiers for tracking
            val_identifiers.append(case_id)

    # Process test files
    print("Processing test files...")
    test_dir = join(input_folder, "test")
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.endswith("_0000.nii.gz")]
        for test_file in test_files:
            # Get case ID from filename
            case_id = test_file[:-len("_0000.nii.gz")]

            # Copy test image
            test_source = join(test_dir, test_file)
            test_target = join(target_imagesTs, test_file)
            copy_with_metadata_conversion(test_source, test_target)

            # Add to test identifiers
            test_identifiers.append(case_id)

    # Create dataset.json - MODIFIED to include all cases in training
    all_identifiers = train_identifiers + val_identifiers
    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = "Pancreatic cancer segmentation and classification"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['channel_names'] = {
        "0": "CT"  # Assuming this is CT data
    }
    
    # Replace the labels section with region-based configuration
    # In prepare_task function, replace the labels part:
    json_dict['labels'] = OrderedDict()  # Use OrderedDict to maintain order
    json_dict['labels']["background"] = 0
    json_dict['labels']["whole_pancreas"] = [1, 2]  # First region (whole pancreas)
    json_dict['labels']["pancreas_lesion"] = 2      # Second region (lesion)
        
    # Add regions_class_order 
    json_dict['regions_class_order'] = [1, 2]  # Place label 1 first, then label 2
    
    # [Rest of the existing code...]
    json_dict['numTraining'] = len(all_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [{'image': f"./imagesTr/{i}_0000.nii.gz", 'label': f"./labelsTr/{i}.nii.gz"} for i in all_identifiers]
    json_dict['test'] = [f"./imagesTs/{i}_0000.nii.gz" for i in test_identifiers]
    json_dict['file_ending'] = file_ending

    save_json(json_dict, join(target_base, "dataset.json"), sort_keys=False)

    # SAVE original train/val splits for later use
    split_info = {
        "training_identifiers": train_identifiers,
        "validation_identifiers": val_identifiers
    }
    save_json(split_info, join(target_base, "validation_identifiers.json"))

    # Print dataset statistics
    print(f"\nDataset prepared successfully at {target_base}")
    print(f"Training cases: {len(train_identifiers)}")
    print(f"Validation cases: {len(val_identifiers)} (added to training folder)")
    print(f"Combined in training folder: {len(all_identifiers)}")
    print(f"Test cases: {len(test_identifiers)}")

    # Print subtype distribution
    train_subtype_counts = {'subtype0': 0, 'subtype1': 0, 'subtype2': 0}
    val_subtype_counts = {'subtype0': 0, 'subtype1': 0, 'subtype2': 0}

    for case_id in train_identifiers:
        subtype = int(case_id.split('_')[1])
        train_subtype_counts[f'subtype{subtype}'] += 1

    for case_id in val_identifiers:
        subtype = int(case_id.split('_')[1])
        val_subtype_counts[f'subtype{subtype}'] += 1

    print("\nSubtype distribution in training set:")
    for subtype, count in train_subtype_counts.items():
        print(f"  {subtype}: {count}")

    print("\nSubtype distribution in validation set:")
    for subtype, count in val_subtype_counts.items():
        print(f"  {subtype}: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert pancreatic cancer dataset to nnUNetv2 format.')
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='Input folder containing the pancreatic cancer dataset')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Output folder where the dataset will be stored in nnUNetv2 format')
    parser.add_argument('--task_id', type=int, default=900,
                        help='Task ID for the dataset (default: 900)')
    parser.add_argument('--task_name', type=str, default='PancreasCancer',
                        help='Task name for the dataset (default: PancreasCancer)')

    args = parser.parse_args()

    prepare_task(args.input_folder, args.output_folder, args.task_id, args.task_name)