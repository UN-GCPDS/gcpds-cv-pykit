import gc
import torch
import random
import matplotlib.pyplot as plt
from typing import Optional

def random_sample_visualization(
    dataset,
    num_classes: int,
    single_class: Optional[int] = None,
    max_classes_to_show: int = 7
) -> None:
    """
    Visualize a random sample from a dataset batch, showing the image and its segmentation masks.

    Args:
        dataset: A PyTorch DataLoader or Dataset that yields (images, masks) batches.
        num_classes (int): Total number of segmentation classes.
        single_class (Optional[int]): If set, visualize only this class mask.
        max_classes_to_show (int): Maximum number of class masks to display at once.

    Returns:
        None. Displays a matplotlib figure.
    """
    # Get a batch from the dataset
    data_iter = iter(dataset)
    images, masks = next(data_iter)
    print(f"Images in the batch: {images.shape}, Masks in the batch: {masks.shape}")

    # Select a random sample within the batch
    sample_idx = random.randint(0, images.shape[0] - 1)

    # Determine number of columns for the visualization grid
    if single_class is not None:
        classes_list = [0]
        n_cols = 2  # Image + single mask
    else:
        # Randomly select up to max_classes_to_show unique classes
        available_classes = list(range(num_classes))
        n_classes_to_show = min(num_classes, max_classes_to_show)
        classes_list = random.sample(available_classes, n_classes_to_show)
        n_cols = 1 + n_classes_to_show  # Image + masks

    n_rows = 1  # Only one sample visualized

    # Create figure and axes
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4 * n_cols, 5),
        squeeze=False
    )

    # Show the original image
    axes[0][0].set_title('Image', loc='center')
    img = images[sample_idx]
    if img.shape[0] == 1:  # Grayscale
        axes[0][0].imshow(img.squeeze(0), cmap='gray')
    else:
        axes[0][0].imshow(img.permute(1, 2, 0).cpu().numpy())

    # Show the masks
    for i, class_idx in enumerate(classes_list):
        ax = axes[0][i + 1]
        ax.set_title(f'Class {class_idx} Mask', loc='center')
        mask_img = masks[sample_idx, class_idx, :, :].cpu().numpy()
        ax.imshow(mask_img, vmin=0.0, vmax=1.0, cmap='gray')
        ax.axis('off')

    # Hide unused axes if any
    for j in range(n_cols):
        axes[0][j].axis('off')
    axes[0][0].axis('off') # Hide the first axis if it's not used

    # Add a super title
    if single_class is not None:
        fig.suptitle(f"Image and Mask for Single Class {single_class}", fontsize=16)
    else:
        fig.suptitle("Image and Segmentation Masks for Random Classes", fontsize=16)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

    # Free memory
    del images, masks
    gc.collect()