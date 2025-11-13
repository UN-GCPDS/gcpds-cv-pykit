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


class AnnotHarmonyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading image patches with multiple annotator masks
    and optional ground truth masks for crowdsourced segmentation tasks.

    This dataset handles loading and preprocessing of images along with annotations
    from multiple annotators. It supports data augmentation, probabilistic mode
    (annotations only), and flexible mask configurations.

    Attributes:
        data_dir (Path): Root directory of the dataset
        image_size (Tuple[int, int]): Target size for images and masks (height, width)
        num_classes (int): Total number of segmentation classes
        num_annotators (int): Number of annotators in the dataset
        partition (str): Dataset partition ('Train', 'Val', 'Test', etc.)
        probabilistic (bool): If True, only load annotator masks (no ground truth)
        annotators (bool): Whether to load annotator masks
        ground_truth (bool): Whether to load ground truth masks
        single_class (Optional[int]): If specified, only loads masks for this class
        augment (bool): Whether to apply data augmentation (only for training)
        ignored_value (float): Fill value for missing annotator masks
        images_folder (str): Name of the folder containing images
        patch_files (List[str]): List of image file paths
        file_sample (List[str]): List of image filenames
        num_samples (int): Total number of samples in the dataset
        masks_path (List[List[str]]): Nested list of annotator mask paths
        ground_truth_masks_path (List[List[Path]]): Nested list of ground truth mask paths

    Args:
        data_dir (Union[str, Path]): Root directory of the dataset
        image_size (Tuple[int, int]): Desired (height, width) for images and masks
        num_classes (int): Number of segmentation classes
        num_annotators (int): Number of annotators
        partition (str): Dataset partition, e.g., 'Train', 'Val', 'Test'
        annotators (bool, optional): Whether to include annotator masks. Defaults to True.
        ground_truth (bool, optional): Whether to include ground truth masks. Defaults to True.
        probabilistic (bool, optional): If True, only load images and annotator masks
            (no ground truth). Defaults to False.
        single_class (Optional[int], optional): If set, only loads ground truth masks
            for this class. Defaults to None.
        images_folder (Optional[str], optional): Name of the folder containing images.
            Defaults to 'patches'.
        augment (bool, optional): Whether to apply data augmentation. Only applied
            during training. Defaults to True.
        ignored_value (float, optional): Fill value for missing annotator masks.
            Defaults to 0.6.

    Raises:
        ValueError: If both annotators and ground_truth are False
        FileNotFoundError: If the specified data directory or partition doesn't exist

    Example:
        >>> dataset = AnnotHarmonyDataset(
        ...     data_dir="/path/to/data",
        ...     image_size=(256, 256),
        ...     num_classes=3,
        ...     num_annotators=5,
        ...     partition="Train",
        ...     augment=True
        ... )
        >>> image, masks, anns_onehot, ground_truth = dataset[0]
        >>> print(f"Image: {image.shape}, Masks: {masks.shape}, GT: {ground_truth.shape}")
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        image_size: Tuple[int, int],
        num_classes: int,
        num_annotators: int,
        partition: str,
        annotators: bool = True,
        ground_truth: bool = True,
        probabilistic: bool = False,
        single_class: Optional[int] = None,
        images_folder: Optional[str] = None,
        augment: bool = True,
        ignored_value: float = 0.6,
    ) -> None:
        """
        Initialize the dataset, collect image and mask file paths, and prepare for loading.

        Args:
            data_dir (Union[str, Path]): Root directory of the dataset.
            image_size (Tuple[int, int]): Desired (height, width) for images and masks.
            num_classes (int): Number of segmentation classes.
            num_annotators (int): Number of annotators.
            partition (str): Dataset partition, e.g., 'Train', 'Val', 'Test'.
            annotators (bool): Whether to include annotator masks.
            ground_truth (bool): Whether to include ground truth masks.
            probabilistic (bool): If True, only load images and annotator masks.
            single_class (Optional[int]): If set, only loads ground truth for this class.
            images_folder (Optional[str]): Name of the folder containing images.
            augment (bool): Whether to apply data augmentation (only in training).
            ignored_value (float): Fill value for missing annotator masks.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.partition = partition
        self.probabilistic = probabilistic
        self.single_class = single_class
        self.augment = augment and (partition.lower() == 'train')
        self.ignored_value = ignored_value
        self.images_folder = images_folder if isinstance(images_folder, str) else 'images'

        # If probabilistic mode is enabled, override ground_truth to False
        if self.probabilistic:
            self.annotators = True
            self.ground_truth = False
        else:
            self.annotators = annotators
            self.ground_truth = ground_truth

        # Validate configuration
        if not self.annotators and not self.ground_truth:
            raise ValueError("At least one of annotators or ground_truth must be True")

        # Find all patch image files - support multiple image formats
        supported_formats = ['*.png', '*.jpg', '*.jpeg']
        self.patch_files = []

        for format_pattern in supported_formats:
            patch_path_pattern = self.data_dir / self.partition / self.images_folder / format_pattern
            format_files = glob.glob(str(patch_path_pattern))
            self.patch_files.extend(format_files)

        # Sort all files together using alphanumeric sorting
        self.patch_files = sorted(self.patch_files, key=self._alphanumeric_key)
        self.file_sample = [Path(f).name for f in self.patch_files]
        self.num_samples = len(self.patch_files)

        print(f"Searching for images in: {self.data_dir / self.partition / self.images_folder}")
        print(f"Supported formats: {', '.join(supported_formats)}")
        print(f"Number of image files found: {self.num_samples}")
        if self.probabilistic:
            print(f"Probabilistic mode enabled - ground truth will not be loaded")

        # Prepare annotator mask paths
        mask_path_main = self.data_dir / self.partition / 'masks'
        list_annotators = [
            ann for ann in os.listdir(mask_path_main)
            if os.path.isdir(mask_path_main / ann) and ann != 'ground_truth'
        ]
        list_annotators = sorted(list_annotators)  # Ensure consistent ordering

        self.masks_path: List[List[str]] = []
        if self.annotators:
            for sample in tqdm(self.file_sample, desc="Organizing annotator masks"):
                masks_sample = []
                for class_id in range(self.num_classes):
                    for annotator in list_annotators:
                        mask_path = mask_path_main / annotator / f'class_{class_id}' / sample
                        masks_sample.append(str(mask_path))
                self.masks_path.append(masks_sample)

        # Prepare ground truth mask paths (skip if probabilistic mode)
        self.ground_truth_masks_path: List[List[Path]] = []
        if self.ground_truth:
            gt_path_main = mask_path_main / 'ground_truth'
            for sample in tqdm(self.file_sample, desc="Organizing ground truth masks"):
                masks_sample = []
                class_ids = [self.single_class] if self.single_class is not None else list(range(self.num_classes))
                for class_id in class_ids:
                    mask_path = gt_path_main / f'class_{class_id}' / sample
                    masks_sample.append(mask_path)
                self.ground_truth_masks_path.append(masks_sample)

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
        if img.float().max() > 1.0:
            img = img.float() / 255.0
        else:
            img = img.float()
        return img

    def process_annotator_masks(
        self, mask_paths: Sequence[Union[str, Path]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read and preprocess annotator mask files for all classes and annotators.

        Args:
            mask_paths (Sequence[Union[str, Path]]): List of mask file paths organized by
                class and annotator.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - masks: Tensor of shape (num_annotators * num_classes, H, W) with annotator masks
                - anns_onehot: Binary tensor of shape (num_annotators,) indicating which
                  annotators provided valid annotations
        """
        masks = torch.zeros(
            self.num_annotators * self.num_classes,
            *self.image_size,
            dtype=torch.float32,
        )
        anns_onehot = [0] * self.num_annotators

        for i, file_path in enumerate(mask_paths):
            if Path(file_path).exists():
                mask = torchvision.io.read_image(str(file_path), mode=ImageReadMode.GRAY)
                mask = TF.resize(mask, list(self.image_size))
                if mask.float().max() > 1.0:
                    mask = mask.float() / 255.0
                else:
                    mask = mask.float()
            else:
                # If mask file does not exist, fill with ignored value
                mask = torch.full((1, *self.image_size), self.ignored_value, dtype=torch.float32)

            annotator_idx = i % self.num_annotators
            if torch.all(mask != self.ignored_value) and anns_onehot[annotator_idx] == 0:
                anns_onehot[annotator_idx] = 1

            masks[i, ...] = mask.squeeze(0)

        return masks, torch.tensor(anns_onehot, dtype=torch.float32)

    def process_ground_truth(self, mask_paths: Sequence[Union[str, Path]]) -> torch.Tensor:
        """
        Read and preprocess ground truth mask files for all classes.

        Args:
            mask_paths (Sequence[Union[str, Path]]): List of ground truth mask file paths
                for each class.

        Returns:
            torch.Tensor: Tensor of shape (C, H, W) with binary ground truth masks for each class.
        """
        num_classes = 1 if self.single_class is not None else self.num_classes
        ground_truth = torch.zeros(num_classes, *self.image_size, dtype=torch.float32)

        for i, file_path in enumerate(mask_paths):
            if Path(file_path).exists():
                mask = torchvision.io.read_image(str(file_path), mode=ImageReadMode.GRAY)
                mask = TF.resize(mask, list(self.image_size))
                if mask.max() > 1.0:
                    mask = mask.float() / 255.0
                else:
                    mask = mask.float()
                ground_truth[i, ...] = mask.squeeze(0)
            else:
                # If mask file does not exist, leave as zeros
                pass

        return ground_truth

    def apply_augmentation(
        self,
        image: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply the same geometric and color augmentations to image and masks.

        Args:
            image (torch.Tensor): Image tensor of shape (3, H, W).
            masks (Optional[torch.Tensor]): Annotator mask tensor of shape
                (num_annotators * num_classes, H, W).
            ground_truth (Optional[torch.Tensor]): Ground truth mask tensor of shape (C, H, W).

        Returns:
            Tuple[torch.Tensor, ...]: Augmented tensors (image, masks, ground_truth) where
                applicable based on which masks are provided.
        """
        # Geometric augmentations (applied to all)
        if random.random() > 0.5:
            image = TF.hflip(image)
            if masks is not None:
                masks = TF.hflip(masks)
            if ground_truth is not None:
                ground_truth = TF.hflip(ground_truth)

        if random.random() > 0.5:
            image = TF.vflip(image)
            if masks is not None:
                masks = TF.vflip(masks)
            if ground_truth is not None:
                ground_truth = TF.vflip(ground_truth)

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            if masks is not None:
                masks = TF.rotate(masks, angle).float()
            if ground_truth is not None:
                ground_truth = TF.rotate(ground_truth, angle).float()

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

        # Return appropriate tuple based on what was provided
        result = [image]
        if masks is not None:
            result.append(masks)
        if ground_truth is not None:
            result.append(ground_truth)

        return tuple(result)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Retrieve the image and mask tensors for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, ...]: Depending on configuration:
                - If annotators and ground_truth: (image, masks, anns_onehot, ground_truth)
                - If annotators only: (image, masks, anns_onehot)
                - If ground_truth only: (image, ground_truth)
        """
        image = self.process_image(self.patch_files[idx])

        if self.annotators and self.ground_truth:
            masks, anns_onehot = self.process_annotator_masks(self.masks_path[idx])
            ground_truth = self.process_ground_truth(self.ground_truth_masks_path[idx])
            if self.augment:
                image, masks, ground_truth = self.apply_augmentation(image, masks, ground_truth)
            return image, masks, anns_onehot, ground_truth

        elif self.annotators:
            masks, anns_onehot = self.process_annotator_masks(self.masks_path[idx])
            if self.augment:
                image, masks = self.apply_augmentation(image, masks, None)
            return image, masks, anns_onehot

        elif self.ground_truth:
            ground_truth = self.process_ground_truth(self.ground_truth_masks_path[idx])
            if self.augment:
                image, ground_truth = self.apply_augmentation(image, None, ground_truth)
            return image, ground_truth

        else:
            raise ValueError("At least one of annotators or ground_truth must be True")


def AnnotHarmonyDataloader(
    data_dir: Union[str, Path],
    batch_size: int,
    image_size: Tuple[int, int],
    num_classes: int,
    num_annotators: int,
    partition: str,
    annotators: bool = True,
    ground_truth: bool = True,
    probabilistic: bool = False,
    single_class: Optional[int] = None,
    augment: bool = True,
    images_folder: Optional[str] = None,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for the AnnotHarmonyDataset.

    Args:
        data_dir (Union[str, Path]): Root directory of the dataset.
        batch_size (int): Batch size.
        image_size (Tuple[int, int]): (height, width) for images and masks.
        num_classes (int): Number of segmentation classes.
        num_annotators (int): Number of annotators.
        partition (str): Dataset partition, e.g., 'Train', 'Val', 'Test'.
        annotators (bool): Whether to include annotator masks.
        ground_truth (bool): Whether to include ground truth masks.
        probabilistic (bool): If True, only load images and annotator masks.
        single_class (Optional[int]): If set, only loads ground truth for this class.
        augment (bool): Whether to apply data augmentation (only in training).
        images_folder (Optional[str]): Name of the folder containing images.
        num_workers (int): Number of worker processes for data loading.
        prefetch_factor (int): Number of batches loaded in advance by each worker.
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = AnnotHarmonyDataset(
        data_dir=data_dir,
        image_size=image_size,
        num_classes=num_classes,
        num_annotators=num_annotators,
        partition=partition,
        annotators=annotators,
        ground_truth=ground_truth,
        probabilistic=probabilistic,
        single_class=single_class,
        augment=augment,
        images_folder=images_folder,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(partition.lower() == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )