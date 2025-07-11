import os
import re
import glob
import torch
import random
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Tuple, Optional, Sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.io
from torchvision.io import ImageReadMode
import torchvision.transforms.functional as TF

class Segmentation_Dataset(Dataset):
    """
    Custom Dataset for semantic segmentation with support for multi-class and single-class masks.

    Args:
        data_dir (Union[str, Path]): Root directory of the dataset.
        image_size (Tuple[int, int]): Desired (height, width) for images and masks.
        num_classes (int): Number of segmentation classes.
        partition (str): Dataset partition, e.g., 'Train', 'Val', 'Test'.
        single_class (Optional[int]): If set, only loads masks for this class.
        augment (bool): Whether to apply data augmentation (only in training).
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        image_size: Tuple[int, int],
        num_classes: int,
        partition: str,
        single_class: Optional[int] = None,
        augment: bool = True
    ) -> None:
        """
        Initialize the dataset, collect image and mask file paths, and prepare for loading.

        Args:
            data_dir (Union[str, Path]): Root directory of the dataset.
            image_size (Tuple[int, int]): Desired (height, width) for images and masks.
            num_classes (int): Number of segmentation classes.
            partition (str): Dataset partition, e.g., 'Train', 'Val', 'Test'.
            single_class (Optional[int]): If set, only loads masks for this class.
            augment (bool): Whether to apply data augmentation (only in training).
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.partition = partition
        self.single_class = single_class
        self.augment = augment and (partition.lower() == 'train')

        # Find all patch image files
        patch_path_pattern = self.data_dir / self.partition / 'patches' / '*.png'
        self.patch_files = sorted(
            glob.glob(str(patch_path_pattern)),
            key=self._alphanumeric_key
        )
        self.file_sample = [Path(f).name for f in self.patch_files]
        self.num_samples = len(self.patch_files)

        print(f"Complete path for patches: {patch_path_pattern}")
        print(f"Number of patch files found: {self.num_samples}")

        # Prepare mask paths for each sample
        mask_path_main = self.data_dir / self.partition / 'masks' / 'ground_truth'
        self.path_masks: List[List[Path]] = []
        for sample in tqdm(self.file_sample, desc="Organizing masks"):
            masks_sample = []
            class_ids = [self.single_class] if self.single_class is not None else list(range(self.num_classes))
            for class_id in class_ids:
                mask_path = mask_path_main / f'class_{class_id}' / sample
                masks_sample.append(mask_path)
            self.path_masks.append(masks_sample)

    @staticmethod
    def _alphanumeric_key(s: str) -> List[Union[int, str]]:
        """
        Generate a key for natural sorting of filenames.

        Args:
            s (str): The string (filename) to split.

        Returns:
            List[Union[int, str]]: List of string and integer parts for sorting.
        """
        return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', s)]

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.num_samples

    def process_image(self, file_path: Sequence[Union[str, Path]]) -> torch.Tensor:
        """
        Read and preprocess an image file.

        Args:
            file_path (Sequence[Union[str, Path]]): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape (3, H, W), normalized to [0, 1].
        """
        img = torchvision.io.read_image(str(file_path), mode=ImageReadMode.RGB)
        img = TF.resize(img, list(self.image_size))
        img = img.float() / 255.0
        return img

    def process_mask(self, mask_paths: Sequence[Union[str, Path]]) -> torch.Tensor:
        """
        Read and preprocess mask files for all classes for a single sample.

        Args:
            mask_paths (Sequence[Union[str, Path]]): List of mask file paths for each class.

        Returns:
            torch.Tensor: Tensor of shape (C, H, W) with binary masks for each class.
        """
        num_classes = 1 if self.single_class is not None else self.num_classes
        ground_truth = torch.zeros(num_classes, *self.image_size, dtype=torch.float32)
        for i, file_path in enumerate(mask_paths):
            if Path(file_path).exists():
                mask = torchvision.io.read_image(str(file_path), mode=ImageReadMode.GRAY)
                mask = TF.resize(mask, list(self.image_size))
                mask = mask.float() 
                ground_truth[i, ...] = mask
            else:
                # If mask file does not exist, leave as zeros
                pass

        return ground_truth

    def apply_augmentation(
        self, image: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the same geometric and color augmentations to image and masks.

        Args:
            image (torch.Tensor): Image tensor of shape (3, H, W).
            masks (torch.Tensor): Mask tensor of shape (C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Augmented image and mask tensors.
        """
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = TF.hflip(masks)
        if random.random() > 0.5:
            image = TF.vflip(image)
            masks = TF.vflip(masks)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            masks = TF.rotate(masks, angle)
            masks = (masks > 0.5).float()
        # Color augmentations (image only)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        if random.random() > 0.7:
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0, 1)
        return image, masks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the image and mask tensors for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor (3, H, W) and mask tensor (C, H, W).
        """
        image = self.process_image(self.patch_files[idx])
        masks = self.process_mask(self.path_masks[idx])
        if self.augment:
            image, masks = self.apply_augmentation(image, masks)
        return image, masks


def Segmentation_DataLoader(
    data_dir: Union[str, Path],
    batch_size: int,
    image_size: Tuple[int, int],
    num_classes: int,
    partition: str,
    single_class: Optional[int] = None,
    augment: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = True
) -> DataLoader:
    """
    Returns a DataLoader for the Segmentation_Dataset.

    Args:
        data_dir (Union[str, Path]): Root directory of the dataset.
        batch_size (int): Batch size.
        image_size (Tuple[int, int]): (height, width) for images and masks.
        num_classes (int): Number of segmentation classes.
        partition (str): Dataset partition, e.g., 'Train', 'Val', 'Test'.
        single_class (Optional[int]): If set, only loads masks for this class.
        augment (bool): Whether to apply data augmentation (only in training).
        num_workers (int): Number of worker processes for data loading.
        prefetch_factor (int): Number of batches loaded in advance by each worker.
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = Segmentation_Dataset(
        data_dir=data_dir,
        image_size=image_size,
        num_classes=num_classes,
        partition=partition,
        single_class=single_class,
        augment=augment
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(partition.lower() == 'Train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0)
    )