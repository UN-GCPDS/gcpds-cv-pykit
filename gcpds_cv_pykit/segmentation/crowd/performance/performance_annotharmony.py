import os
import gc
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

import torch
import numpy as np
from torch.amp import autocast

from ..losses import TGCE_SS


def calculate_metrics_probabilistic(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ignore_value: float = 0.6,
    smooth: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate segmentation metrics for probabilistic mode by averaging over multiple thresholds.
    
    Args:
        y_pred (torch.Tensor): Prediction tensor.
        y_true (torch.Tensor): Probability mask (averaged annotations).
        thresholds (List[float]): List of thresholds to evaluate.
        ignore_value (float): Value in y_true to ignore.
        smooth (float): Smoothing factor.
        
    Returns:
        Tuple containing average and per-class metrics.
    """
    num_thresholds = len(thresholds)
    
    # Initialize accumulators
    dice_avg_sum = 0.0
    jaccard_avg_sum = 0.0
    sensitivity_avg_sum = 0.0
    specificity_avg_sum = 0.0
    
    num_classes = y_pred.shape[1]
    dice_per_class_sum = torch.zeros(num_classes, device=y_pred.device)
    jaccard_per_class_sum = torch.zeros(num_classes, device=y_pred.device)
    sensitivity_per_class_sum = torch.zeros(num_classes, device=y_pred.device)
    specificity_per_class_sum = torch.zeros(num_classes, device=y_pred.device)
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        metrics = calculate_metrics(y_pred, y_true, threshold, ignore_value, smooth)
        
        dice_avg_sum += metrics[0]
        jaccard_avg_sum += metrics[1]
        sensitivity_avg_sum += metrics[2]
        specificity_avg_sum += metrics[3]
        dice_per_class_sum += metrics[4]
        jaccard_per_class_sum += metrics[5]
        sensitivity_per_class_sum += metrics[6]
        specificity_per_class_sum += metrics[7]
    
    # Average over all thresholds
    dice_avg = dice_avg_sum / num_thresholds
    jaccard_avg = jaccard_avg_sum / num_thresholds
    sensitivity_avg = sensitivity_avg_sum / num_thresholds
    specificity_avg = specificity_avg_sum / num_thresholds
    dice_per_class = dice_per_class_sum / num_thresholds
    jaccard_per_class = jaccard_per_class_sum / num_thresholds
    sensitivity_per_class = sensitivity_per_class_sum / num_thresholds
    specificity_per_class = specificity_per_class_sum / num_thresholds
    
    return (
        dice_avg, jaccard_avg, sensitivity_avg, specificity_avg,
        dice_per_class, jaccard_per_class, sensitivity_per_class, specificity_per_class
    )

def compute_probability_mask(
    masks: torch.Tensor,
    num_annotators: int,
    num_classes: int,
    ignore_value: float = 0.6
) -> torch.Tensor:
    """
    Compute probability mask from annotator masks.
    
    The mask structure is: [ann_1_class_0, ..., ann_m_class_0, ..., ann_1_class_n, ..., ann_m_class_n]
    where m is the number of annotators and n is the number of classes.
    
    Args:
        masks (torch.Tensor): Annotator masks with shape [B, m*n, H, W]
        num_annotators (int): Number of annotators
        num_classes (int): Number of classes
        ignore_value (float): Value to ignore in masks
        
    Returns:
        torch.Tensor: Probability mask with shape [B, n, H, W]
    """
    batch_size = masks.shape[0]
    
    # Reshape masks to [B, num_classes, num_annotators, H, W]
    masks_reshaped = masks.view(batch_size, num_classes, num_annotators, *masks.shape[2:])
    
    # Create mask to ignore specified values
    valid_mask = (masks_reshaped != ignore_value).float()
    
    # Count valid annotations per pixel per class
    valid_count = torch.sum(valid_mask, dim=2)  # [B, num_classes, H, W]
    
    # Sum valid annotations
    masks_sum = torch.sum(masks_reshaped * valid_mask, dim=2)  # [B, num_classes, H, W]
    
    # Compute average (probability), avoiding division by zero
    probability_mask = torch.where(
        valid_count > 0,
        masks_sum / valid_count,
        torch.zeros_like(masks_sum)
    )
    
    return probability_mask


def calculate_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
    ignore_value: float = 0.6,
    smooth: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate segmentation metrics, supporting per-class and average metrics.

    Args:
        y_pred (torch.Tensor): Prediction tensor.
        y_true (torch.Tensor): Ground truth tensor.
        threshold (float): Threshold for binarization.
        ignore_value (float): Value in y_true to ignore.
        smooth (float): Smoothing factor.

    Returns:
        Tuple containing average and per-class metrics.
    """
    
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Create mask to ignore specified values
    ignore_tensor = torch.tensor(ignore_value, device=y_true.device)
    mask = (y_true != ignore_tensor).float()
    
    # Convert to binary masks using threshold
    y_true = (y_true > threshold).float()
    y_pred = (y_pred > threshold).float()
    
    # Calculate confusion matrix components
    tp = torch.sum(y_true * y_pred * mask, dim=(2, 3))  # True Positives
    fp = torch.sum((1 - y_true) * y_pred * mask, dim=(2, 3))  # False Positives
    fn = torch.sum(y_true * (1 - y_pred) * mask, dim=(2, 3))  # False Negatives
    tn = torch.sum((1 - y_true) * (1 - y_pred) * mask, dim=(2, 3))  # True Negatives
    
    # Calculate metrics per sample and class
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    jaccard = (tp + smooth) / (tp + fp + fn + smooth)
    sensitivity = (tp + smooth) / (tp + fn + smooth)  # Recall/True Positive Rate
    specificity = (tn + smooth) / (tn + fp + smooth)  # True Negative Rate
    
    # Handle NaN values by replacing with 0.0
    dice = torch.where(torch.isnan(dice), torch.tensor(0.0, device=dice.device), dice)
    jaccard = torch.where(torch.isnan(jaccard), torch.tensor(0.0, device=jaccard.device), jaccard)
    sensitivity = torch.where(torch.isnan(sensitivity), torch.tensor(0.0, device=sensitivity.device), sensitivity)
    specificity = torch.where(torch.isnan(specificity), torch.tensor(0.0, device=specificity.device), specificity)
    
    # Calculate averages across batch and classes
    dice_avg = torch.mean(dice)
    jaccard_avg = torch.mean(jaccard)
    sensitivity_avg = torch.mean(sensitivity)
    specificity_avg = torch.mean(specificity)
    
    # Calculate per-class averages (averaged across batch dimension)
    dice_per_class = torch.mean(dice, dim=0)
    jaccard_per_class = torch.mean(jaccard, dim=0)
    sensitivity_per_class = torch.mean(sensitivity, dim=0)
    specificity_per_class = torch.mean(specificity, dim=0)
    
    return (
        dice_avg, jaccard_avg, sensitivity_avg, specificity_avg,
        dice_per_class, jaccard_per_class, sensitivity_per_class, specificity_per_class
    )

def PerformanceAnnotHarmony(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    save_results: bool = False,
    probabilistic: bool = False,
    thresholds: Optional[List[float]] = None
) -> None:
    """
    Evaluate a segmentation model with TGCE_SS loss on a test dataset.
    
    Args:
        model: PyTorch model to evaluate
        test_dataset: DataLoader containing test data
        config: Configuration dictionary with keys like:
            - "Num of annotators" (int)
            - "Number of classes" (int)
            - "Single class test" (int or None)
            - "AMixPre" (bool, use mixed precision)
            - "Main_model", "Dataset" (for saving results)
            - "drive_dir" (str, optional save directory)
            - "Ignored value" (float, default 0.6)
            - "Q paramater" (float, default 0.7)
            - "Smooth" (float, default 1e-7)
        save_results: Whether to save evaluation results to disk
        probabilistic: Whether to use probabilistic approach (evaluate against averaged annotations)
        thresholds: List of thresholds for probabilistic evaluation (default: [0.1, 0.2, ..., 0.9])
    """
    device = torch.device(config.get("Device", "cuda:0"))
    model.to(device)
    model.eval()

    # Loss function (TGCE for annotator harmony training)
    loss_fn = TGCE_SS(
        annotators=config["Num of annotators"],
        classes=config["Number of classes"],
        ignore_value=config.get("Ignored value", 0.6),
        q=config.get("Q paramater", 0.7),
    )

    smooth = config.get("Smooth", 1e-7)
    ignore_value = config.get("Ignored value", 0.6)
    
    # Default thresholds for probabilistic approach
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Storage for results
    loss_results: List[float] = []
    dice_results: List[float] = []
    jaccard_results: List[float] = []
    sensitivity_results: List[float] = []
    specificity_results: List[float] = []

    is_single_class = isinstance(config.get("Single class test"), int)
    num_classes = 1 if is_single_class else config["Number of classes"]

    dice_per_class = [[] for _ in range(num_classes)]
    jaccard_per_class = [[] for _ in range(num_classes)]
    sensitivity_per_class = [[] for _ in range(num_classes)]
    specificity_per_class = [[] for _ in range(num_classes)]

    for data_batch in tqdm(test_dataset, desc="Testing model's performance"):
        # Unpack based on whether we have ground truth or not
        if len(data_batch) == 4:
            # Standard mode: (images, masks, annotations, gt_masks)
            images, masks, anns_onehot, gt_masks = [x.to(device) for x in data_batch]
        elif len(data_batch) == 3:
            # Probabilistic mode without ground truth: (images, masks, annotations)
            images, masks, anns_onehot = [x.to(device) for x in data_batch]
            gt_masks = None
        else:
            raise ValueError(f"Unexpected number of elements in data_batch: {len(data_batch)}")

        with torch.no_grad():
            if config.get("AMixPre", False):
                with autocast(device_type=device.type):
                    y_pred = model(images, anns_onehot)
                    loss = loss_fn(y_pred, masks)
            else:
                y_pred = model(images, anns_onehot)
                loss = loss_fn(y_pred, masks)

            # Safe loss
            loss = loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)

            # Class selection
            if is_single_class:
                class_idx = config["Single class test"]
                y_pred_seg = y_pred[0][:, class_idx:class_idx+1]
            else:
                y_pred_seg = y_pred[0][:, :config["Number of classes"]]

            # Determine ground truth or probability mask
            if probabilistic or gt_masks is None:
                # Compute probability mask from annotator masks
                y_true = compute_probability_mask(
                    masks, 
                    config["Num of annotators"], 
                    config["Number of classes"],
                    ignore_value
                )
                
                # Select class if needed
                if is_single_class:
                    class_idx = config["Single class test"]
                    y_true = y_true[:, class_idx:class_idx+1]
                
                # Calculate metrics across multiple thresholds
                metrics = calculate_metrics_probabilistic(
                    y_pred_seg, 
                    y_true, 
                    thresholds, 
                    ignore_value, 
                    smooth
                )
            else:
                # Use ground truth masks
                y_true = gt_masks.float()
                
                # Calculate metrics with standard threshold
                metrics = calculate_metrics(
                    y_pred_seg, 
                    y_true, 
                    threshold=0.5, 
                    ignore_value=ignore_value, 
                    smooth=smooth
                )

            # Unpack metrics
            dice_avg, jaccard_avg, sensitivity_avg, specificity_avg = metrics[:4]
            dice_per_class_batch, jaccard_per_class_batch = metrics[4:6]
            sensitivity_per_class_batch, specificity_per_class_batch = metrics[6:8]

            # Store per-class metrics
            for c in range(num_classes):
                dice_per_class[c].append(dice_per_class_batch[c].item())
                jaccard_per_class[c].append(jaccard_per_class_batch[c].item())
                sensitivity_per_class[c].append(sensitivity_per_class_batch[c].item())
                specificity_per_class[c].append(specificity_per_class_batch[c].item())

            # Store global metrics
            loss_results.append(loss.item())
            dice_results.append(dice_avg.item() if torch.is_tensor(dice_avg) else dice_avg)
            jaccard_results.append(jaccard_avg.item() if torch.is_tensor(jaccard_avg) else jaccard_avg)
            sensitivity_results.append(sensitivity_avg.item() if torch.is_tensor(sensitivity_avg) else sensitivity_avg)
            specificity_results.append(specificity_avg.item() if torch.is_tensor(specificity_avg) else specificity_avg)

    # Convert results to numpy
    loss_results = np.array(loss_results)
    dice_results = np.array(dice_results)
    jaccard_results = np.array(jaccard_results)
    sensitivity_results = np.array(sensitivity_results)
    specificity_results = np.array(specificity_results)

    dice_per_class = [np.array(x) for x in dice_per_class]
    jaccard_per_class = [np.array(x) for x in jaccard_per_class]
    sensitivity_per_class = [np.array(x) for x in sensitivity_per_class]
    specificity_per_class = [np.array(x) for x in specificity_per_class]

    # Print metrics
    mode_label = " (Probabilistic Mode)" if probabilistic else ""
    print(f"\nGlobal Performance Metrics{mode_label}:")
    print(f"Loss mean: {loss_results.mean():.5f}, std: {loss_results.std():.5f}")
    print(f"Dice mean: {dice_results.mean():.5f}, std: {dice_results.std():.5f}")
    print(f"Jaccard mean: {jaccard_results.mean():.5f}, std: {jaccard_results.std():.5f}")
    print(f"Sensitivity mean: {sensitivity_results.mean():.5f}, std: {sensitivity_results.std():.5f}")
    print(f"Specificity mean: {specificity_results.mean():.5f}, std: {specificity_results.std():.5f}")

    print(f"\nPer-Class Performance Metrics{mode_label}:")
    for c in range(num_classes):
        print(f"\nClass {c}:")
        print(f"Dice mean: {dice_per_class[c].mean():.5f}, std: {dice_per_class[c].std():.5f}")
        print(f"Jaccard mean: {jaccard_per_class[c].mean():.5f}, std: {jaccard_per_class[c].std():.5f}")
        print(f"Sensitivity mean: {sensitivity_per_class[c].mean():.5f}, std: {sensitivity_per_class[c].std():.5f}")
        print(f"Specificity mean: {specificity_per_class[c].mean():.5f}, std: {specificity_per_class[c].std():.5f}")

    # Save results if required
    if save_results:
        drive_dir = config.get("drive_dir", ".")
        os.makedirs(f"{drive_dir}/results", exist_ok=True)

        prob_suffix = "_probabilistic" if probabilistic else ""
        filename_base = f"{drive_dir}/results/{config.get('Main_model','')}_{config.get('Dataset','')}{prob_suffix}"
        
        np.save(f"{filename_base}_Loss.npy", loss_results)
        np.save(f"{filename_base}_Dice_global.npy", dice_results)
        np.save(f"{filename_base}_Jaccard_global.npy", jaccard_results)
        np.save(f"{filename_base}_Sensitivity_global.npy", sensitivity_results)
        np.save(f"{filename_base}_Specificity_global.npy", specificity_results)

        for c in range(num_classes):
            np.save(f"{filename_base}_Dice_class{c}.npy", dice_per_class[c])
            np.save(f"{filename_base}_Jaccard_class{c}.npy", jaccard_per_class[c])
            np.save(f"{filename_base}_Sensitivity_class{c}.npy", sensitivity_per_class[c])
            np.save(f"{filename_base}_Specificity_class{c}.npy", specificity_per_class[c])

        print(f"\nResults saved to: {filename_base}_*.npy")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()





