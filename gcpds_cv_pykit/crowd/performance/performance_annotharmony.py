import os
import gc
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

import torch
import numpy as np
from torch.amp import autocast

from ..losses import TGCE_SS


def PerformanceAnnotHarmony(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    save_results: bool = False
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
        save_results: Whether to save evaluation results to disk
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
        # Unpack: (images, masks, annotations, gt_masks)
        images, masks, anns_onehot, gt_masks = [x.to(device) for x in data_batch]

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
                y_pred = y_pred[:, class_idx:class_idx+1]
            else:
                y_pred = y_pred[:, :config["Number of classes"]]

            y_true = gt_masks.float()
            y_pred = y_pred.float()

            # Ignore mask
            ignore_value = torch.tensor(0.6, device=device)
            mask = (y_true != ignore_value).float()

            # Binarize predictions
            y_pred = (y_pred > 0.5).float()

            # Confusion matrix terms
            intersection = torch.sum(y_true * y_pred * mask, dim=(2, 3))
            sum_true = torch.sum(y_true * mask, dim=(2, 3))
            sum_pred = torch.sum(y_pred * mask, dim=(2, 3))

            # Metrics per class
            dice_batch = (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth)
            jaccard_batch = (intersection + smooth) / (sum_true + sum_pred - intersection + smooth)
            sensitivity_batch = (intersection + smooth) / (sum_true + smooth)
            specificity_batch = (
                torch.sum((1 - y_true) * (1 - y_pred) * mask, dim=(2, 3)) + smooth
            ) / (torch.sum((1 - y_true) * mask, dim=(2, 3)) + smooth)

            # Global averages (NaN-safe)
            dice = torch.nan_to_num(torch.mean(dice_batch, dim=0))
            jaccard = torch.nan_to_num(torch.mean(jaccard_batch, dim=0))
            sensitivity = torch.nan_to_num(torch.mean(sensitivity_batch, dim=0))
            specificity = torch.nan_to_num(torch.mean(specificity_batch, dim=0))

            # Store per-class metrics
            for c in range(num_classes):
                dice_per_class[c].append(dice_batch[:, c].mean().item())
                jaccard_per_class[c].append(jaccard_batch[:, c].mean().item())
                sensitivity_per_class[c].append(sensitivity_batch[:, c].mean().item())
                specificity_per_class[c].append(specificity_batch[:, c].mean().item())

            # Store global metrics
            loss_results.append(loss.item())
            dice_results.append(dice.mean().item())
            jaccard_results.append(jaccard.mean().item())
            sensitivity_results.append(sensitivity.mean().item())
            specificity_results.append(specificity.mean().item())

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
    print("\nGlobal Performance Metrics:")
    print(f"Loss mean: {loss_results.mean():.5f}, std: {loss_results.std():.5f}")
    print(f"Dice mean: {dice_results.mean():.5f}, std: {dice_results.std():.5f}")
    print(f"Jaccard mean: {jaccard_results.mean():.5f}, std: {jaccard_results.std():.5f}")
    print(f"Sensitivity mean: {sensitivity_results.mean():.5f}, std: {sensitivity_results.std():.5f}")
    print(f"Specificity mean: {specificity_results.mean():.5f}, std: {specificity_results.std():.5f}")

    print("\nPer-Class Performance Metrics:")
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

        filename_base = f"{drive_dir}/results/{config.get('Main_model','')}_{config.get('Dataset','')}"
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

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
