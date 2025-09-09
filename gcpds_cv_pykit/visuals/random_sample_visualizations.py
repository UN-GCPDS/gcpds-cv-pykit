import gc
import torch
import random
import matplotlib.pyplot as plt
from typing import Optional, Union, Iterator, Tuple
from torch.utils.data import DataLoader


def random_sample_visualization(
    dataset: Union[DataLoader, Iterator[Tuple[torch.Tensor, ...]]],
    num_classes: int,
    single_class: Optional[int] = None,
    max_classes_to_show: int = 7,
    type: Optional[str] = None,
    num_annotators: Optional[int] = None,
    annotators: bool = False,
    ground_truth: bool = False
) -> None:
    """
    Visualize a random sample from a dataset with its corresponding segmentation masks.

    Supports two visualization modes:
    - type="baseline": Show the image and randomly selected segmentation masks.
    - type="annot_harmony": Show image, annotators' masks, and optionally ground truth masks.

    Args:
        dataset (Union[DataLoader, Iterator]): PyTorch DataLoader or iterator yielding batches.
        num_classes (int): Number of segmentation classes.
        single_class (Optional[int]): If provided, only show this class mask.
        max_classes_to_show (int): Max number of classes to display in baseline mode.
        type (Optional[str]): Visualization mode. Supports "baseline" and "annot_harmony".
        num_annotators (Optional[int]): Number of annotators (required if type="annot_harmony").
        annotators (bool): If True, display annotators' masks (only for type="annot_harmony").
        ground_truth (bool): If True, display ground truth masks (only for type="annot_harmony").

    Raises:
        ValueError: If invalid arguments are provided.
    """

    # -------------------------------------------------------------------------
    # BASELINE MODE
    # -------------------------------------------------------------------------
    if type == "baseline":
        if max_classes_to_show <= 0:
            raise ValueError("max_classes_to_show must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        data_iter = iter(dataset)
        images, masks = next(data_iter)

        print(f"Images: {images.shape}, Masks: {masks.shape}")

        if images.shape[0] == 0:
            raise IndexError("Batch is empty")

        sample_idx = random.randint(0, images.shape[0] - 1)

        if single_class is not None:
            classes_list = [single_class]
            n_cols = 2
        else:
            available_classes = list(range(num_classes))
            n_classes_to_show = min(num_classes, max_classes_to_show)
            classes_list = random.sample(available_classes, n_classes_to_show)
            n_cols = 1 + n_classes_to_show

        fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 5))

        axes[0].set_title("Image", loc="center")
        img = images[sample_idx]
        if img.shape[0] == 1:  # grayscale
            axes[0].imshow(img.squeeze(0).cpu().numpy(), cmap="viridis")
        else:  # rgb
            display_img = img[:3] if img.shape[0] >= 3 else img
            axes[0].imshow(display_img.permute(1, 2, 0).cpu().numpy())
        axes[0].axis("off")

        for i, class_idx in enumerate(classes_list):
            ax = axes[i + 1]
            ax.set_title(f"Class {class_idx}", loc="center")
            ax.imshow(masks[sample_idx, class_idx].cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")

        fig.suptitle(
            f"Image and Mask for Class {single_class}" if single_class is not None
            else "Image and Random Segmentation Masks",
            fontsize=16
        )
        fig.tight_layout()
        plt.show()

        del images, masks
        gc.collect()

    # -------------------------------------------------------------------------
    # ANNOT_HARMONY MODE
    # -------------------------------------------------------------------------
    elif type == "annot_harmony":
        if num_annotators is None:
            raise ValueError("num_annotators must be specified when type='annot_harmony'")
        if not (annotators or ground_truth):
            raise ValueError("At least one of annotators or ground_truth must be True")

        data_iter = iter(dataset)
        first_batch = next(data_iter)

        # Unpack batch depending on presence of ground truth
        if annotators and ground_truth:
            images, anns_masks, anns_onehot, gt_masks = first_batch
            print(f"Images: {images.shape}, AnnsMasks: {anns_masks.shape}, "
                  f"AnnsOneHot: {anns_onehot.shape}, GT: {gt_masks.shape}")
        elif annotators and not ground_truth:
            images, anns_masks, anns_onehot = first_batch
            gt_masks = None
            print(f"Images: {images.shape}, AnnsMasks: {anns_masks.shape}, AnnsOneHot: {anns_onehot.shape}")
        elif not annotators and ground_truth:
            images, gt_masks = first_batch
            anns_masks, anns_onehot = None, None
            print(f"Images: {images.shape}, GT: {gt_masks.shape}")
        else:
            raise ValueError("Invalid configuration in annot_harmony")

        sample_idx = random.randint(0, images.shape[0] - 1)

        rows, cols = 1, [1]
        row_annot = None
        row_gt = None

        if annotators and anns_masks is not None:
            cols.append(min(7, anns_masks.shape[1] // num_classes))
            row_annot = rows
            rows += 1
        if ground_truth and gt_masks is not None:
            cols.append(min(7, gt_masks.shape[1]))
            row_gt = rows
            rows += 1
        cols = int(max(cols))

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(rows + 1, cols, hspace=0.2, wspace=0.2)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(rows)]

        # Show image
        axes[0][0].set_title("Image", loc="center")
        axes[0][0].imshow(images[sample_idx].permute(1, 2, 0))
        axes[0][0].axis("off")

        # Annotators masks
        if annotators and row_annot is not None and anns_masks is not None:
            title_ax = fig.add_subplot(gs[row_annot, :])
            title_ax.set_title("Annotators' Segmentation Masks", loc="center")
            title_ax.axis("off")

            chosen_ann = random.sample(range(num_annotators), min(num_annotators, cols))
            chosen_classes = [random.randint(0, num_classes - 1) for _ in chosen_ann]

            for i, ann_idx in enumerate(chosen_ann):
                mask_index = ann_idx + chosen_classes[i] * num_annotators
                axes[row_annot][i].imshow(anns_masks[sample_idx, mask_index], cmap="viridis", vmin=0, vmax=1)
                axes[row_annot][i].axis("off")

        # Ground truth masks
        if ground_truth and row_gt is not None and gt_masks is not None:
            if single_class is None:
                chosen_classes = random.sample(range(num_classes), min(num_classes, cols))
                title_ax = fig.add_subplot(gs[row_gt, :])
                title_ax.set_title("Ground Truth Segmentation Masks", loc="center")
                title_ax.axis("off")
            else:
                chosen_classes = [single_class]
                axes[row_gt][0].set_title(f"GT Class {single_class}", loc="center")

            for i, class_idx in enumerate(chosen_classes):
                axes[row_gt][i].imshow(gt_masks[sample_idx, class_idx], cmap="viridis", vmin=0, vmax=1)
                axes[row_gt][i].axis("off")

        [ax.axis('off') for row in axes for ax in row]
        fig.tight_layout()
        plt.show()

        # Cleanup
        del images
        if annotators:
            del anns_masks, anns_onehot
        if ground_truth:
            del gt_masks
        gc.collect()

    else:
        raise ValueError(f"Invalid type: {type}. Supported: 'baseline', 'annot_harmony'")
