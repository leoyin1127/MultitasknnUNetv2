#!/usr/bin/env python3
"""
Validation script to evaluate the improved inference pipeline
by comparing segmentation and classification results against ground truth.
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import csv
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, load_json, maybe_mkdir_p
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import glob

def dice_coefficient(y_true, y_pred):
    """Calculate Dice coefficient"""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

def evaluate_segmentation(gt_folder, pred_folder):
    """Evaluate segmentation performance"""
    # Find all ground truth segmentation files (those without _0000 in the name)
    gt_files = []
    
    # Handle subtype directory structure
    subtype_dirs = [d for d in os.listdir(gt_folder) if os.path.isdir(join(gt_folder, d)) and 'subtype' in d]
    
    if subtype_dirs:
        print(f"Found {len(subtype_dirs)} subtype directories")
        for subtype_dir in subtype_dirs:
            subtype_path = join(gt_folder, subtype_dir)
            # Find segmentation files (no _0000 in name)
            for file in os.listdir(subtype_path):
                if file.endswith('.nii.gz') and '_0000' not in file:
                    gt_files.append((join(subtype_path, file), file))
    else:
        # No subtype directories, look for files directly in gt_folder
        for file in os.listdir(gt_folder):
            if file.endswith('.nii.gz') and '_0000' not in file:
                gt_files.append((join(gt_folder, file), file))
    
    print(f"Found {len(gt_files)} ground truth segmentation files")
    
    whole_pancreas_dices = []
    lesion_dices = []
    
    for gt_path, gt_file in gt_files:
        # Check if corresponding prediction exists
        pred_file = gt_file
        pred_path = join(pred_folder, pred_file)
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file {pred_file} not found, skipping")
            continue
            
        # Load ground truth
        gt_img = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_img)
        
        # Load prediction
        pred_img = sitk.ReadImage(pred_path)
        pred_array = sitk.GetArrayFromImage(pred_img)
        
        # Calculate Dice for whole pancreas (label 1 + label 2)
        gt_whole = (gt_array > 0).astype(np.uint8)
        pred_whole = (pred_array > 0).astype(np.uint8)
        whole_dice = dice_coefficient(gt_whole, pred_whole)
        whole_pancreas_dices.append(whole_dice)
        
        # Calculate Dice for lesion (label 2)
        gt_lesion = (gt_array == 2).astype(np.uint8)
        pred_lesion = (pred_array == 2).astype(np.uint8)
        lesion_dice = dice_coefficient(gt_lesion, pred_lesion)
        lesion_dices.append(lesion_dice)
        
        print(f"File: {gt_file}")
        print(f"  Whole pancreas Dice: {whole_dice:.4f}")
        print(f"  Lesion Dice: {lesion_dice:.4f}")
    
    # Calculate average Dice scores
    avg_whole_dice = np.mean(whole_pancreas_dices) if whole_pancreas_dices else 0
    avg_lesion_dice = np.mean(lesion_dices) if lesion_dices else 0
    
    return {
        "whole_pancreas_dice": avg_whole_dice,
        "lesion_dice": avg_lesion_dice,
        "individual_whole_dices": whole_pancreas_dices,
        "individual_lesion_dices": lesion_dices
    }

def evaluate_classification(gt_folder, pred_csv):
    """Evaluate classification performance"""
    # Check if prediction CSV exists
    if not os.path.exists(pred_csv):
        print(f"Error: Prediction CSV file {pred_csv} not found")
        return {
            "accuracy": 0,
            "f1_score": 0,
            "confusion_matrix": np.zeros((3, 3)),
            "num_samples": 0
        }
    
    # Extract ground truth classes from folder structure
    gt_classes = {}
    
    # Handle subtype directories
    subtype_dirs = [d for d in os.listdir(gt_folder) if os.path.isdir(join(gt_folder, d)) and 'subtype' in d]
    
    if subtype_dirs:
        for subtype_dir in subtype_dirs:
            subtype_path = join(gt_folder, subtype_dir)
            subtype_num = int(subtype_dir[-1])  # Extract subtype (0, 1, 2)
            
            # Find image files in this subtype (those with _0000 in name)
            for file in os.listdir(subtype_path):
                if file.endswith('.nii.gz'):
                    if '_0000' in file:
                        # This is an image file
                        case_id = file.replace('_0000.nii.gz', '')
                        gt_classes[f"{case_id}.nii.gz"] = subtype_num
                    elif '_' not in file:
                        # This is a segmentation file
                        case_id = file.replace('.nii.gz', '')
                        gt_classes[f"{case_id}.nii.gz"] = subtype_num
    
    print(f"Found ground truth classes for {len(gt_classes)} cases")
    
    # Load predictions from CSV
    pred_classes = {}
    with open(pred_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                pred_classes[row[0]] = int(row[1])
    
    print(f"Loaded {len(pred_classes)} predictions from CSV")
    
    # Match ground truth with predictions
    y_true = []
    y_pred = []
    matched_cases = []
    
    for case_name, true_class in gt_classes.items():
        if case_name in pred_classes:
            y_true.append(true_class)
            y_pred.append(pred_classes[case_name])
            matched_cases.append(case_name)
    
    print(f"Matched {len(matched_cases)} cases between ground truth and predictions")
    
    # Calculate metrics
    if len(y_true) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "confusion_matrix": cm,
            "num_samples": len(y_true)
        }
    else:
        print("No matching cases found between ground truth and predictions")
        return {
            "accuracy": 0,
            "f1_score": 0,
            "confusion_matrix": np.zeros((3, 3)),
            "num_samples": 0
        }

def visualize_results(seg_metrics, cls_metrics, output_folder):
    """Create visualizations of the evaluation results"""
    maybe_mkdir_p(output_folder)
    
    # Segmentation visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot individual Dice scores if available
    if len(seg_metrics["individual_whole_dices"]) > 0:
        ax[0].boxplot([seg_metrics["individual_whole_dices"], seg_metrics["individual_lesion_dices"]], 
                    labels=["Whole Pancreas", "Lesion"])
        ax[0].set_ylabel("Dice Score")
        ax[0].set_title("Segmentation Performance")
        ax[0].grid(True, alpha=0.3)
        
        # Add target lines
        ax[0].axhline(y=0.91, color='r', linestyle='--', alpha=0.5, label="Target Whole Pancreas (0.91)")
        ax[0].axhline(y=0.31, color='g', linestyle='--', alpha=0.5, label="Target Lesion (0.31)")
        ax[0].legend()
    else:
        ax[0].text(0.5, 0.5, "No segmentation data available", ha='center', va='center')
        ax[0].set_title("Segmentation Performance")
    
    # Plot confusion matrix if available
    if cls_metrics["num_samples"] > 0:
        im = ax[1].imshow(cls_metrics["confusion_matrix"], interpolation='nearest', cmap=plt.cm.Blues)
        ax[1].set_title(f"Confusion Matrix\nF1={cls_metrics['f1_score']:.4f}, Acc={cls_metrics['accuracy']:.4f}")
        ax[1].set_xticks([0, 1, 2])
        ax[1].set_yticks([0, 1, 2])
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("True")
        
        # Add text annotations to confusion matrix
        for i in range(3):
            for j in range(3):
                text = ax[1].text(j, i, cls_metrics["confusion_matrix"][i, j],
                            ha="center", va="center", color="white" if cls_metrics["confusion_matrix"][i, j] > 3 else "black")
    else:
        ax[1].text(0.5, 0.5, "No classification data available", ha='center', va='center')
        ax[1].set_title("Classification Performance")
    
    plt.tight_layout()
    plt.savefig(join(output_folder, "validation_results.png"))
    
    # Save summary as text
    with open(join(output_folder, "validation_summary.txt"), 'w') as f:
        f.write("======= Validation Results =======\n\n")
        f.write("Segmentation Performance:\n")
        f.write(f"  Whole Pancreas Dice: {seg_metrics['whole_pancreas_dice']:.4f} (Target: 0.91+)\n")
        f.write(f"  Lesion Dice: {seg_metrics['lesion_dice']:.4f} (Target: 0.31+)\n\n")
        
        f.write("Classification Performance:\n")
        f.write(f"  F1 Score: {cls_metrics['f1_score']:.4f} (Target: 0.70+)\n")
        f.write(f"  Accuracy: {cls_metrics['accuracy']:.4f}\n")
        f.write(f"  Samples: {cls_metrics['num_samples']}\n\n")
        
        f.write("Confusion Matrix:\n")
        for i in range(3):
            f.write(f"  {cls_metrics['confusion_matrix'][i]}\n")
    
    print(f"Visualization saved to {output_folder}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate improvements in segmentation and classification performance")
    parser.add_argument("--validation_data", type=str, required=True, 
                        help="Folder containing validation data with ground truth")
    parser.add_argument("--prediction_folder", type=str, required=True,
                        help="Folder containing segmentation predictions")
    parser.add_argument("--prediction_csv", type=str, required=True,
                        help="CSV file with classification predictions")
    parser.add_argument("--output_folder", type=str, default="./validation_results",
                        help="Folder to save validation results")
    
    args = parser.parse_args()
    
    print("Starting validation of improvements...")
    
    # Evaluate segmentation
    print("\nEvaluating segmentation performance...")
    seg_metrics = evaluate_segmentation(args.validation_data, args.prediction_folder)
    
    print(f"\nAverage Whole Pancreas Dice: {seg_metrics['whole_pancreas_dice']:.4f}")
    print(f"Average Lesion Dice: {seg_metrics['lesion_dice']:.4f}")
    
    # Evaluate classification
    print("\nEvaluating classification performance...")
    cls_metrics = evaluate_classification(args.validation_data, args.prediction_csv)
    
    print(f"Classification Accuracy: {cls_metrics['accuracy']:.4f}")
    print(f"Classification F1 Score: {cls_metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(cls_metrics["confusion_matrix"])
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(seg_metrics, cls_metrics, args.output_folder)
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()