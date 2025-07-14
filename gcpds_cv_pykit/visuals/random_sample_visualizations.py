import gc
import torch
import random
import matplotlib.pyplot as plt
from typing import Optional, Union, Iterator, Tuple
from torch.utils.data import DataLoader, Dataset


def random_sample_visualization(
    dataset: Union[DataLoader, Iterator[Tuple[torch.Tensor, torch.Tensor]]],
    num_classes: int,
    single_class: Optional[int] = None,
    max_classes_to_show: int = 7,
    type: Optional[str] = None
) -> None:
    """
    Visualize a random sample from a dataset with its corresponding segmentation masks.
    
    This function displays an image alongside its segmentation masks for visualization
    purposes during dataset exploration or model debugging. It supports both single-class
    and multi-class visualization modes.
    
    Args:
        dataset (Union[DataLoader, Iterator[Tuple[torch.Tensor, torch.Tensor]]]): 
            A PyTorch DataLoader or iterator that yields batches of (images, masks).
            Images should have shape (B, C, H, W) and masks should have shape (B, num_classes, H, W).
        num_classes (int): 
            Total number of classes in the segmentation task. Must be positive.
        single_class (Optional[int], optional): 
            If specified, only visualize the mask for this specific class index.
            Must be in range [0, num_classes-1]. If None, visualizes multiple classes.
            Defaults to None.
        max_classes_to_show (int, optional): 
            Maximum number of classes to display when single_class is None.
            Will randomly sample up to this many classes for visualization.
            Must be positive. Defaults to 7.
        type (Optional[str], optional): 
            Visualization type selector. Currently only supports type=1 for batch visualization.
            If None or any value other than 'baseline', raises ValueError. Defaults to None.
    
    Returns:
        None: This function displays the visualization using matplotlib and doesn't return anything.
    
    Raises:
        ValueError: If type is not 1 or if single_class is out of valid range.
        StopIteration: If the dataset iterator is empty.
        IndexError: If the batch is empty or sample_idx is out of range.
        RuntimeError: If tensor operations fail (e.g., GPU/CPU mismatch).
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> # Assuming you have a segmentation dataset
        >>> train_loader = DataLoader(my_dataset, batch_size=4, shuffle=True)
        >>> 
        >>> # Visualize random classes
        >>> random_sample_visualization(
        ...     dataset=train_loader,
        ...     num_classes=5,
        ...     max_classes_to_show=3,
        ...     type=1
        ... )
        >>> 
        >>> # Visualize only class 2
        >>> random_sample_visualization(
        ...     dataset=train_loader,
        ...     num_classes=5,
        ...     single_class=2,
        ...     type=1
        ... )
    
    Note:
        - Images can be grayscale (C=1) or RGB (C=3)
        - Masks are expected to be binary or probability maps with values in [0, 1]
        - The function automatically handles memory cleanup using garbage collection
        - Visualization uses matplotlib, so ensure you're in an environment that supports it
    """
    
    if type == 'baseline':
        # Validate inputs
    
        if max_classes_to_show <= 0:
            raise ValueError(f"max_classes_to_show must be positive, got {max_classes_to_show}")
        
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        # Get a batch from the dataset
        data_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]] = iter(dataset)
        images: torch.Tensor
        masks: torch.Tensor
        images, masks = next(data_iter)
        
        print(f"Images in the batch: {images.shape}, Masks in the batch: {masks.shape}")

        # Validate batch is not empty
        if images.shape[0] == 0:
            raise IndexError("Batch is empty, cannot select a sample") 

        # Select a random sample within the batch
        sample_idx: int = random.randint(0, images.shape[0] - 1)

        # Determine number of columns for the visualization grid
        classes_list: list[int]
        n_cols: int
        
        if single_class is not None:
            classes_list = [single_class]
            n_cols = 2  # Image + single mask
        else:
            # Randomly select up to max_classes_to_show unique classes
            available_classes: list[int] = list(range(num_classes))
            n_classes_to_show: int = min(num_classes, max_classes_to_show)
            classes_list = random.sample(available_classes, n_classes_to_show)
            n_cols = 1 + n_classes_to_show  # Image + masks

        n_rows: int = 1  # Only one sample visualized

        # Create figure and axes
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(2 * n_cols, 5),
            squeeze=False
        )

        # Show the original image
        axes[0][0].set_title('Image', loc='center')
        img: torch.Tensor = images[sample_idx]
        
        if img.shape[0] == 1:  # Grayscale
            axes[0][0].imshow(img.squeeze(0).cpu().numpy(), cmap='gray')
        else:  # RGB or multi-channel
            # Ensure we only take first 3 channels for RGB display
            display_img = img[:3] if img.shape[0] >= 3 else img
            axes[0][0].imshow(display_img.permute(1, 2, 0).cpu().numpy())

        # Show the masks
        for i, class_idx in enumerate(classes_list):
            ax = axes[0][i + 1]
            ax.set_title(f'Class {class_idx} Mask', loc='center')
            mask_img: torch.Tensor = masks[sample_idx, class_idx, :, :]
            ax.imshow(mask_img.cpu().numpy(), vmin=0.0, vmax=1.0, cmap='gray')
            ax.axis('off')

        # Hide axes for all subplots
        for j in range(n_cols):
            axes[0][j].axis('off')

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
    
    else:
        raise ValueError(
            f"Invalid type specified. Only type='baseline' is supported for random sample visualization, got {type}"
        )