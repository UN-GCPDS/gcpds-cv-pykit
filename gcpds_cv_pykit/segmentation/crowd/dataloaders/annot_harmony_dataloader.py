import os
import re
import glob
import random
import torch
import torchvision.io
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from pathlib import Path
from typing import Union, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.io import ImageReadMode


class AnnotHarmonyDataset(Dataset):
    """
    PyTorch Dataset for loading image patches, annotator masks, 
    and ground truth masks with optional data augmentation.
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
        augment: bool = True,
        ignored_value: float = 0.6,
    ):
        """
        Args:
            data_dir (str | Path): Root dataset directory.
            image_size (Tuple[int, int]): Target image size (H, W).
            num_classes (int): Number of segmentation classes.
            num_annotators (int): Number of annotators.
            partition (str): Dataset split (e.g., 'Train', 'Val', 'Test').
            annotators (bool): Whether to include annotator masks.
            ground_truth (bool): Whether to include ground truth masks.
            probabilistic (bool): If True, only load images and masks (no ground truth).
            single_class (int, optional): If set, only load this class.
            augment (bool): Apply augmentations (only in training).
            ignored_value (float): Fill value for missing masks.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.partition = partition
        self.probabilistic = probabilistic
        
        # If probabilistic mode is enabled, override ground_truth to False
        if self.probabilistic:
            self.annotators = True
            self.ground_truth = False
        else:
            self.annotators = annotators
            self.ground_truth = ground_truth
            
        self.single_class = single_class
        self.augment = augment and (partition.lower() == "train")
        self.ignored_value = ignored_value

        # --- Collect patch files ---
        patch_path_pattern = self.data_dir / partition / "patches" / "*.png"
        patch_files = glob.glob(str(patch_path_pattern))

        self.patch_files = sorted(patch_files, key=self._alphanumeric_key)
        self.file_sample = [Path(f).name for f in self.patch_files]
        self.num_samples = len(self.patch_files)

        print(f"[INFO] Found {self.num_samples} patch files in {patch_path_pattern}")
        if self.probabilistic:
            print(f"[INFO] Probabilistic mode enabled - ground truth will not be loaded")

        # --- Collect annotator masks ---
        mask_path_main = self.data_dir / partition / "masks"
        list_annotators = [
            ann for ann in os.listdir(mask_path_main)
            if os.path.isdir(mask_path_main / ann) and ann != "ground_truth"
        ]

        self.masks_path = []
        for sample in tqdm(self.file_sample, desc="Organizing masks"):
            masks_sample = []
            for class_id in range(self.num_classes):
                for annotator in list_annotators:
                    mask_path = mask_path_main / annotator / f"class_{class_id}" / sample
                    masks_sample.append(str(mask_path))
            self.masks_path.append(masks_sample)

        # --- Collect ground truth masks (skip if probabilistic mode) ---
        if not self.probabilistic:
            gt_path_main = mask_path_main / "ground_truth"
            self.ground_truth_masks_path = []
            for sample in tqdm(self.file_sample, desc="Organizing GT masks"):
                masks_sample = []
                if self.single_class is not None:
                    mask_path = gt_path_main / f"class_{self.single_class}" / sample
                    masks_sample.append(str(mask_path))
                else:
                    for class_id in range(self.num_classes):
                        mask_path = gt_path_main / f"class_{class_id}" / sample
                        masks_sample.append(str(mask_path))
                self.ground_truth_masks_path.append(masks_sample)
        else:
            self.ground_truth_masks_path = []

    @staticmethod
    def _alphanumeric_key(s: str):
        """Sort helper: split string into text + numbers for correct ordering."""
        parts = re.split(r"(\d+)", s)
        return [int(p) if p.isdigit() else p for p in parts]

    def __len__(self) -> int:
        return self.num_samples

    def _process_image(self, file_path: str) -> torch.Tensor:
        """Load and normalize RGB image."""
        img = torchvision.io.decode_image(file_path, mode=ImageReadMode.RGB)
        img = TF.resize(img, list(self.image_size))
        if img.float().max() > 1.0:
            return img.float() / 255.0
        else:
            return img.float()

    def _process_masks(self, mask_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load annotator masks into tensor."""
        masks = torch.zeros(
            self.num_annotators * self.num_classes,
            *self.image_size,
            dtype=torch.float32,
        )
        anns_one_hot = [0] * self.num_annotators

        for i, file_path in enumerate(mask_paths):
            if os.path.exists(file_path):
                mask = torchvision.io.decode_image(file_path, mode=ImageReadMode.GRAY)
                if mask.float().max() > 1.0:
                    mask = mask.float() / 255.0
            else:
                mask = torch.full(self.image_size, self.ignored_value, dtype=torch.float32)

            mask = TF.resize(mask, list(self.image_size)).float()
            annotator_idx = i % self.num_annotators
            if torch.all(mask != self.ignored_value) and anns_one_hot[annotator_idx] == 0:
                anns_one_hot[annotator_idx] = 1
            masks[i, ...] = mask

        return masks, torch.tensor(anns_one_hot, dtype=torch.float32)

    def _process_ground_truth(self, mask_paths: List[str]) -> torch.Tensor:
        """Load ground truth masks into tensor."""
        num_classes = 1 if self.single_class is not None else self.num_classes
        ground_truth = torch.zeros(num_classes, *self.image_size, dtype=torch.float32)

        for i, file_path in enumerate(mask_paths):
            mask = torchvision.io.decode_image(file_path, mode=ImageReadMode.GRAY)
            ground_truth[i, ...] = TF.resize(mask, list(self.image_size)).float()

        if ground_truth.max() > 1.0:
            ground_truth = ground_truth / 255.0
        return ground_truth

    def _apply_augmentation(
        self,
        image: torch.Tensor,
        masks: Optional[torch.Tensor],
        ground_truth: Optional[torch.Tensor],
    ):
        """Apply the same augmentations to image and masks."""
        if random.random() > 0.5:
            image = TF.hflip(image)
            if masks is not None: masks = TF.hflip(masks)
            if ground_truth is not None: ground_truth = TF.hflip(ground_truth)

        if random.random() > 0.5:
            image = TF.vflip(image)
            if masks is not None: masks = TF.vflip(masks)
            if ground_truth is not None: ground_truth = TF.vflip(ground_truth)

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            if masks is not None: masks = TF.rotate(masks, angle).float()
            if ground_truth is not None: ground_truth = TF.rotate(ground_truth, angle).float()

        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        if random.random() > 0.7:
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0, 1)

        if masks is not None and ground_truth is not None:
            return image, masks, ground_truth
        elif masks is not None:
            return image, masks
        elif ground_truth is not None:
            return image, ground_truth
        else:
            raise ValueError("At least one of annotators or ground_truth must be True")

    def __getitem__(self, idx: int):
        """Return samples as tuple (no dicts)."""
        image = self._process_image(self.patch_files[idx])

        if self.annotators and self.ground_truth:
            masks, anns_onehot = self._process_masks(self.masks_path[idx])
            ground_truth = self._process_ground_truth(self.ground_truth_masks_path[idx])
            if self.augment:
                aug_result = self._apply_augmentation(image, masks, ground_truth)
                if len(aug_result) == 3:
                    image, masks, ground_truth = aug_result
                else:
                    image, masks = aug_result
            return image, masks, anns_onehot, ground_truth

        elif self.annotators:
            masks, anns_onehot = self._process_masks(self.masks_path[idx])
            if self.augment:
                aug_result = self._apply_augmentation(image, masks, None)
                image, masks = aug_result[:2]
            return image, masks, anns_onehot

        elif self.ground_truth:
            ground_truth = self._process_ground_truth(self.ground_truth_masks_path[idx])
            if self.augment:
                aug_result = self._apply_augmentation(image, None, ground_truth)
                if len(aug_result) == 3:
                    image, _, ground_truth = aug_result
                else:
                    image, ground_truth = aug_result
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
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Utility function to create DataLoader.
    
    Args:
        probabilistic (bool): If True, only load images and masks (no ground truth).
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
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(partition.lower() == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )