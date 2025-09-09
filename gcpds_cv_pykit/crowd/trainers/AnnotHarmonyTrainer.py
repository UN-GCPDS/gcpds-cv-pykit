import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import os
import gc
import time
import copy
import wandb
import random
import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Union, List, Tuple, Optional, Dict, Any
from ..models import AnnotHarmonyModel
from ..losses import TGCE_SS

class AnnotHarmonyTrainer:
    def __init__(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        config: Dict[str, Any]
    ) -> None:
        
        self.config: Dict[str, Any] = config
        self.train_loader: DataLoader = train_loader
        self.valid_loader: Optional[DataLoader] = valid_loader

    def wandb_logging(self) -> str:
        """
        Configure Weights & Biases logging if enabled.

        Returns:
            str: Status message indicating if W&B tracking is enabled

        Raises:
            ValueError: If wandb_monitoring is not None or a list with exactly three strings
        """
        if self.wandb_monitoring is None:
            self.wandb_monitoring = False
            return "WandB tracking disabled."
        elif (
            isinstance(self.wandb_monitoring, list)
            and len(self.wandb_monitoring) == 3
            and all(isinstance(item, str) for item in self.wandb_monitoring)
        ):
            wandb.login(key=self.wandb_monitoring[0])
            keys_to_include = {
                'Model', 'Number of classes', 'Num of annotators', 
                'Epochs', 'Loss function'
            }
            wandb_config = {key: value for key, value in self.config.items() if key in keys_to_include}
            self.run = wandb.init(
                project=self.wandb_monitoring[1],
                name=self.wandb_monitoring[2],
                config=wandb_config
            )
            self.wandb_monitoring = True
            return "WandB tracking enabled."
        else:
            raise ValueError("wandb_monitoring must be None or a list of exactly three strings.")

    def print_device_info(self) -> None:
        """Print detailed device information including GPU details if available."""
        
        print("=" * 50)
        print("DEVICE INFORMATION")
        print("=" * 50)
        
        # Current device being used
        print(f"Current device: {self.device}")
        print(f"Device type: {self.device.type}")
        
        # CUDA availability and details
        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            
            # Current CUDA device details
            if self.device.type == 'cuda':
                current_device = self.device.index if self.device.index is not None else torch.cuda.current_device()
                print(f"Current CUDA device index: {current_device}")
                print(f"Current CUDA device name: {torch.cuda.get_device_name(current_device)}")
                
                # Memory information
                memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3  # GB
                
                print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB")
                print(f"GPU Memory - Reserved: {memory_reserved:.2f} GB") 
                print(f"GPU Memory - Total: {memory_total:.2f} GB")
                
                # GPU properties
                props = torch.cuda.get_device_properties(current_device)
                print(f"GPU Compute Capability: {props.major}.{props.minor}")
                print(f"GPU Multiprocessors: {props.multi_processor_count}")
            
            # List all available CUDA devices
            if torch.cuda.device_count() > 1:
                print("\nAll available CUDA devices:")
                for i in range(torch.cuda.device_count()):
                    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                    
        else:
            print(f"CUDA available: No")
            print(f"Using CPU for computation")
        
        # PyTorch version
        print(f"PyTorch version: {torch.__version__}")
        
        # Model device (if model exists)
        if hasattr(self, 'model') and self.model is not None:
            model_device = next(self.model.parameters()).device
            print(f"Model is on device: {model_device}")
            
            # Check if model and target device match
            if model_device == self.device:
                print("✓ Model and target device match")
            else:
                print("✗ WARNING: Model and target device mismatch!")
        
        print("=" * 50)

    def model_handling(self) -> nn.Module:
        """Initialize and return the Annotation Harmony model."""
        
        # Import the model class (assuming it's available)
        # from ..models import Annot_Harmony_Model
        
        self.model = AnnotHarmonyModel(
            in_ch=self.config.get("Input size", (3,))[0],
            out_ch=self.config.get('Number of classes', 1),
            n_annotators=self.config.get('Num of annotators', 1),
            activation_seg=self.config.get('Activation seg', 'sparse_softmax'),
            activation_rel=self.config.get('Activation rel', 'softmax')
        )
        
        return self.model

    def loss_handling(self) -> nn.Module:
        """Initialize and return the loss function."""
        
        loss_fn = self.config.get('Loss function', 'TGCE_SS')
        
        match loss_fn:
            case 'TGCE_SS':
                return TGCE_SS(
                    annotators=self.config.get('Num of annotators', 1),
                    classes=self.config.get('Number of classes', 1),
                    ignore_value=self.config.get('Ignore value', 0.6),
                    q=self.config.get('Q parameter', 0.7243854912956864)
                )
            
            case _:
                raise ValueError(f"Unknown loss function: {loss_fn}")

    def training_phases(self, phase: int) -> None:
        """
        Configure training parameters for different phases of training.

        Args:
            phase (int): The training phase (1, 2, 3, 4, or False for no phases).
        """
        match phase:
            case 1:
                params = []
                # Freeze encoder parameters
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                # Enable decoder, seg_head and ann_rel parameters
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.seg_head.parameters():
                    param.requires_grad = True
                for param in self.model.ann_rel.parameters():
                    param.requires_grad = True

                params.extend([
                    {'params': self.model.decoder.parameters(), 'lr': 1e-4},
                    {'params': self.model.seg_head.parameters(), 'lr': 1e-4},
                    {'params': self.model.ann_rel.parameters(), 'lr': 1e-4}
                ])
                self.optimizer = optim.Adam(params)

            case 2:
                # Freeze encoder parameters first
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

                # Enable and collect BatchNorm parameters
                bn_params = []
                for m in self.model.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        bn_params.extend(list(m.parameters()))
                        for param in m.parameters():
                            param.requires_grad = True

                # Enable decoder, seg_head and ann_rel parameters
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.seg_head.parameters():
                    param.requires_grad = True
                for param in self.model.ann_rel.parameters():
                    param.requires_grad = True

                params = [
                    {'params': self.model.decoder.parameters(), 'lr': 1e-4},
                    {'params': self.model.seg_head.parameters(), 'lr': 1e-4},
                    {'params': self.model.ann_rel.parameters(), 'lr': 1e-4},
                    {'params': bn_params, 'lr': 1e-5}
                ]
                self.optimizer = optim.Adam(params)

            case 3:
                # First freeze all encoder parameters
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

                # Create sets to track parameters we've already added
                added_params = set()

                # Enable and collect layer4 parameters first
                layer4_params = list(self.model.encoder.layer4.parameters())
                for param in layer4_params:
                    param.requires_grad = True
                    added_params.add(param)

                # Enable and collect BatchNorm parameters (excluding those in layer4)
                bn_params = []
                for m in self.model.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        for param in m.parameters():
                            if param not in added_params:
                                param.requires_grad = True
                                bn_params.append(param)
                                added_params.add(param)

                # Enable decoder, seg_head and ann_rel parameters
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.seg_head.parameters():
                    param.requires_grad = True
                for param in self.model.ann_rel.parameters():
                    param.requires_grad = True

                params = [
                    {'params': self.model.decoder.parameters(), 'lr': 1e-4},
                    {'params': self.model.seg_head.parameters(), 'lr': 1e-4},
                    {'params': self.model.ann_rel.parameters(), 'lr': 1e-4},
                    {'params': bn_params, 'lr': 1e-5},
                    {'params': layer4_params, 'lr': 1e-5}
                ]
                self.optimizer = optim.Adam(params)

            case 4:
                # First freeze all encoder parameters
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

                # Create sets to track parameters we've already added
                added_params = set()

                # Enable and collect layer3 parameters first
                layer3_params = list(self.model.encoder.layer3.parameters())
                for param in layer3_params:
                    param.requires_grad = True
                    added_params.add(param)

                # Enable and collect layer4 parameters next
                layer4_params = list(self.model.encoder.layer4.parameters())
                for param in layer4_params:
                    param.requires_grad = True
                    added_params.add(param)

                # Enable and collect BatchNorm parameters (excluding those in layer3 and layer4)
                bn_params = []
                for m in self.model.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        for param in m.parameters():
                            if param not in added_params:
                                param.requires_grad = True
                                bn_params.append(param)
                                added_params.add(param)

                # Enable decoder, seg_head and ann_rel parameters
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.seg_head.parameters():
                    param.requires_grad = True
                for param in self.model.ann_rel.parameters():
                    param.requires_grad = True

                params = [
                    {'params': self.model.decoder.parameters(), 'lr': 1e-4},
                    {'params': self.model.seg_head.parameters(), 'lr': 1e-4},
                    {'params': self.model.ann_rel.parameters(), 'lr': 1e-4},
                    {'params': bn_params, 'lr': 1e-5},
                    {'params': layer3_params, 'lr': 1e-5},
                    {'params': layer4_params, 'lr': 1e-5}
                ]
                self.optimizer = optim.Adam(params)
                self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.94)
            
            case False:
                self.optimizer = optim.Adam(self.model.parameters())
                self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.94)

    def perform_across_epochs(self) -> None:
        """Create and save training plots after training completion."""
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Find the next available experiment number
        experiment_num = 1
        while os.path.exists(f"results/experiment_{experiment_num}"):
            experiment_num += 1
        
        # Create the experiment folder
        experiment_folder = f"results/experiment_{experiment_num}"
        os.makedirs(experiment_folder)

        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.epochs), self.train_loss, label='Training Loss', marker='o')
        if self.valid_loader is not None and hasattr(self, 'val_loss'):
            plt.plot(range(self.epochs), self.val_loss, label='Validation Loss', marker='x')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{experiment_folder}/Loss.png')
        plt.close()

        # Only create metric plots if ground truth is available
        if self.train_ground_truth:
            # DICE plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(self.epochs), self.train_global_dice, label='Training Global DICE', marker='o')
            for i in range(self.config['Number of classes']):
                plt.plot(range(self.epochs), self.train_per_class_dice[i], label=f'Training Class {i} DICE', linestyle='--')
            if self.valid_loader is not None and self.valid_ground_truth:
                plt.plot(range(self.epochs), self.val_global_dice, label='Validation Global DICE', marker='x')
                for i in range(self.config['Number of classes']):
                    plt.plot(range(self.epochs), self.val_per_class_dice[i], label=f'Validation Class {i} DICE', linestyle=':')
            plt.title('DICE Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('DICE')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{experiment_folder}/DICE.png')
            plt.close()

            # Similar plots for Jaccard, Sensitivity, Specificity can be added here
            # Following the same pattern as in the reference trainer

    def visualizations(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        anns_onehot: torch.Tensor,
        epoch: int
    ) -> None:
        """Generate and display training visualizations."""

        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.eval()
        sample_n = random.randint(0, images.shape[0] - 1)
        sample = images[sample_n:sample_n + 1].to(self.device)
        anns_onehot_sample = anns_onehot[sample_n:sample_n + 1].to(self.device)
        mask_sample = masks[sample_n:sample_n + 1]

        with torch.no_grad():
            if self.amp_:
                with autocast(self.device.type):
                    prediction = self.model(sample, anns_onehot_sample)
            else:
                prediction = self.model(sample, anns_onehot_sample)

        # Convert tensors to numpy for visualization
        sample_np = sample.cpu().numpy().transpose(0, 2, 3, 1)
        prediction_np = prediction.cpu().numpy()
        mask_sample_np = mask_sample.cpu().numpy()

        # Number of elements to display (maximum 9)
        elements_display = min(prediction_np.shape[1], 9)

        # Create figure and axes
        fig, axs = plt.subplots(4, elements_display, figsize=(16, 5))

        # Display input image
        axs[0, 0].imshow(sample_np[0])
        axs[0, 0].set_title('Input image')

        # Display class predictions
        axs[1, 0].imshow(np.argmax(prediction_np, axis=1)[0], vmin=0.0, vmax=self.config['Number of classes']-1)
        axs[1, 0].set_title(f'Predictions class')

        # Display annotator reliability maps
        for idx, ann_rel in enumerate(random.sample(range(self.config['Num of annotators']), 
                                                   min(elements_display, self.config['Num of annotators']))):
            axs[2, idx].imshow(prediction_np[0, ann_rel + self.config['Number of classes']], vmin=0.0, vmax=1.0)
            axs[2, idx].set_title(f"Ann {ann_rel + 1}'s rel map")

        for idx, anns in enumerate(random.sample(range(self.config['Num of annotators']), 
                                                min(elements_display, self.config['Num of annotators']))):
            annotatios_ann = [anns + self.config['Num of annotators'] * class_ for class_ in range(self.config['Number of classes'])]
            if np.all(mask_sample_np[:, annotatios_ann, :, :] == self.config.get('Ignore value', 0.6)):
                mask_show = np.full(mask_sample_np.shape[2:], -1.0)  # Initialize with ignore value
            else:
                mask_show = np.argmax(mask_sample_np[:, annotatios_ann, :, :], axis=1)[0]
            axs[3, idx].imshow(mask_show, vmin=-1.0, vmax=self.config['Number of classes']-1)
            axs[3, idx].set_title(f"Ann {anns + 1}'s masks")

        [ax.axis('off') for row in axs for ax in row]
        plt.tight_layout()
        plt.show()
        plt.close()

        if self.wandb_monitoring:
            self.run.log({f"Predictions_Epoch_{epoch}": wandb.Image(fig)})

    def calculate_metrics(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor, 
        threshold: float = 0.5, 
        ignore_value: float = 0.6, 
        smooth: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def train_step(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        anns_onehot: torch.Tensor,
        orig_mask: Optional[torch.Tensor] = None
    ) -> Union[float, Tuple[float, ...]]:
        """
        Execute a single training step on the current batch.

        Args:
            images (torch.Tensor): Batch of input images
            masks (torch.Tensor): Batch of annotator masks
            anns_onehot (torch.Tensor): One-hot encoded annotator information
            orig_mask (torch.Tensor, optional): Batch of ground truth masks

        Returns:
            If train_ground_truth is True: tuple of (loss, metrics...)
            Otherwise: loss value
        """

        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model or optimizer not initialized.")

        self.model.train()
        images = images.to(self.device)
        masks = masks.to(self.device)
        anns_onehot = anns_onehot.to(self.device)
        
        if orig_mask is not None:
            orig_mask = orig_mask.to(self.device)

        self.optimizer.zero_grad()

        if self.amp_:
            if self.scaler is None:
                raise RuntimeError("AMP is enabled but scaler is not initialized.")
            with autocast(self.device.type):
                y_pred = self.model(images, anns_onehot)
                loss = self.loss_fn(y_pred, masks)
            if not torch.isnan(loss):
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            y_pred = self.model(images, anns_onehot)
            loss = self.loss_fn(y_pred, masks)
            if not torch.isnan(loss):
                loss.backward()
                self.optimizer.step()

        # Calculate metrics if ground truth is available
        if self.train_ground_truth and orig_mask is not None:
            if isinstance(self.single_class_train, int):
                selected_pred = y_pred[:, self.single_class_train:self.single_class_train + 1]
                metrics = self.calculate_metrics(selected_pred, orig_mask)
            else:
                selected_pred = y_pred[:, :self.config['Number of classes']]
                metrics = self.calculate_metrics(selected_pred, orig_mask)
            return (loss.item(), *metrics)
        else:
            return loss.item()

    def val_step(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        anns_onehot: Optional[torch.Tensor] = None,
        orig_mask: Optional[torch.Tensor] = None
    ) -> Union[float, Tuple[float, ...]]:
        """
        Execute a single validation step on the current batch.

        Args:
            images (torch.Tensor): Batch of input images
            masks (torch.Tensor, optional): Batch of annotator masks
            anns_onehot (torch.Tensor, optional): One-hot encoded annotator information
            orig_mask (torch.Tensor, optional): Batch of ground truth masks

        Returns:
            Validation metrics based on configuration
        """

        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.eval()
        images = images.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        if anns_onehot is not None:
            anns_onehot = anns_onehot.to(self.device)
        if orig_mask is not None:
            orig_mask = orig_mask.to(self.device)

        with torch.no_grad():
            if self.amp_:
                with autocast(self.device.type):
                    y_pred = self.model(images, anns_onehot)
                    if self.annotators_valid and masks is not None:
                        loss = self.loss_fn(y_pred, masks)
                        loss = loss.item() if not torch.isnan(loss) else 0.0
            else:
                y_pred = self.model(images, anns_onehot)
                if self.annotators_valid and masks is not None:
                    loss = self.loss_fn(y_pred, masks)
                    loss = loss.item() if not torch.isnan(loss) else 0.0

            # Calculate metrics if ground truth is available
            if self.valid_ground_truth and orig_mask is not None:
                if isinstance(self.single_class_valid, int):
                    selected_pred = y_pred[:, self.single_class_valid:self.single_class_valid + 1]
                    metrics = self.calculate_metrics(selected_pred, orig_mask)
                else:
                    selected_pred = y_pred[:, :self.config['Number of classes']]
                    metrics = self.calculate_metrics(selected_pred, orig_mask)

            # Return appropriate values based on configuration
            if self.annotators_valid and self.valid_ground_truth:
                return (loss, *metrics)
            elif self.annotators_valid and not self.valid_ground_truth:
                return loss
            elif not self.annotators_valid and self.valid_ground_truth:
                return metrics
            else:
                raise ValueError("Invalid validation data configuration")

    def training(self) -> None:
        """Execute the complete training procedure."""

        self.best_train_dice = 0.0
        self.best_train_loss = float('inf')
        self.best_val_dice = 0.0
        self.best_val_loss = float('inf')

        self.train_loss = np.zeros(self.epochs)
        self.val_loss = np.zeros(self.epochs)

        # Initialize metric arrays if ground truth is available
        if self.train_ground_truth:
            self.train_global_dice = np.zeros(self.epochs)
            self.train_per_class_dice = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]
            self.train_global_jaccard = np.zeros(self.epochs)
            self.train_per_class_jaccard = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]
            self.train_global_sensitivity = np.zeros(self.epochs)
            self.train_per_class_sensitivity = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]
            self.train_global_specificity = np.zeros(self.epochs)
            self.train_per_class_specificity = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]

        if self.valid_ground_truth:
            self.val_global_dice = np.zeros(self.epochs)
            self.val_per_class_dice = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]
            self.val_global_jaccard = np.zeros(self.epochs)
            self.val_per_class_jaccard = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]
            self.val_global_sensitivity = np.zeros(self.epochs)
            self.val_per_class_sensitivity = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]
            self.val_global_specificity = np.zeros(self.epochs) 
            self.val_per_class_specificity = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]

        self.start_time = time.time()

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            if self.train_phases:
                match epoch:
                    case 0:
                        print("Training phase 1")
                        self.training_phases(1)
                    case 10:
                        print("Training phase 2")
                        self.training_phases(2)
                    case 20:
                        print("Training phase 3")
                        self.training_phases(3)
                    case 30:
                        print("Training phase 4")
                        self.training_phases(4)
                    case _:
                        pass
            else:
                self.training_phases(False)

            # Training loop
            total_train_loss = 0.0
            num_train_batches = 0
            total_train_dice_avg = 0.0
            total_train_jaccard_avg = 0.0
            total_train_sensitivity_avg = 0.0
            total_train_specificity_avg = 0.0

            num_classes = self.config['Number of classes']
            total_train_dice_per_class = torch.zeros(num_classes)
            total_train_jaccard_per_class = torch.zeros(num_classes)
            total_train_sensitivity_per_class = torch.zeros(num_classes)
            total_train_specificity_per_class = torch.zeros(num_classes)

            for data_batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}/{self.epochs}"):
                if self.train_ground_truth and len(data_batch) == 4:
                    images, masks, anns_onehot, orig_mask = data_batch
                    train_metrics = self.train_step(images, masks, anns_onehot, orig_mask)
                    
                    batch_loss = train_metrics[0]
                    batch_dice_avg = train_metrics[1].cpu()
                    batch_jaccard_avg = train_metrics[2].cpu()
                    batch_sensitivity_avg = train_metrics[3].cpu()
                    batch_specificity_avg = train_metrics[4].cpu()
                    batch_dice_per_class = train_metrics[5].cpu()
                    batch_jaccard_per_class = train_metrics[6].cpu()
                    batch_sensitivity_per_class = train_metrics[7].cpu()
                    batch_specificity_per_class = train_metrics[8].cpu()

                    total_train_loss += batch_loss
                    total_train_dice_avg += batch_dice_avg
                    total_train_jaccard_avg += batch_jaccard_avg
                    total_train_sensitivity_avg += batch_sensitivity_avg
                    total_train_specificity_avg += batch_specificity_avg
                    total_train_dice_per_class += batch_dice_per_class
                    total_train_jaccard_per_class += batch_jaccard_per_class
                    total_train_sensitivity_per_class += batch_sensitivity_per_class
                    total_train_specificity_per_class += batch_specificity_per_class
                else:
                    images, masks, anns_onehot = data_batch
                    batch_loss = self.train_step(images, masks, anns_onehot)
                    total_train_loss += batch_loss

                num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches

            if self.train_ground_truth:
                avg_train_dice = total_train_dice_avg / num_train_batches
                avg_train_jaccard = total_train_jaccard_avg / num_train_batches
                avg_train_sensitivity = total_train_sensitivity_avg / num_train_batches
                avg_train_specificity = total_train_specificity_avg / num_train_batches
                avg_train_dice_per_class = total_train_dice_per_class / num_train_batches
                avg_train_jaccard_per_class = total_train_jaccard_per_class / num_train_batches
                avg_train_sensitivity_per_class = total_train_sensitivity_per_class / num_train_batches
                avg_train_specificity_per_class = total_train_specificity_per_class / num_train_batches

            if self.train_phases:
                if epoch > 30:
                    self.scheduler.step()
            else:
                self.scheduler.step()

            # Validation loop
            total_val_loss = 0.0
            num_val_batches = 0
            total_val_dice_avg = 0.0
            total_val_jaccard_avg = 0.0
            total_val_sensitivity_avg = 0.0
            total_val_specificity_avg = 0.0
            total_val_dice_per_class = torch.zeros(num_classes)
            total_val_jaccard_per_class = torch.zeros(num_classes)
            total_val_sensitivity_per_class = torch.zeros(num_classes)
            total_val_specificity_per_class = torch.zeros(num_classes)

            if self.valid_loader is not None:
                for data_batch in tqdm(self.valid_loader, desc=f"Validation Epoch {epoch + 1}/{self.epochs}"):
                    if self.annotators_valid and self.valid_ground_truth and len(data_batch) == 4:
                        images, masks, anns_onehot, orig_mask = data_batch
                        val_metrics = self.val_step(images, masks, anns_onehot, orig_mask)
                        
                        batch_loss = val_metrics[0]
                        batch_dice_avg = val_metrics[1].cpu()
                        batch_jaccard_avg = val_metrics[2].cpu()
                        batch_sensitivity_avg = val_metrics[3].cpu()
                        batch_specificity_avg = val_metrics[4].cpu()
                        batch_dice_per_class = val_metrics[5].cpu()
                        batch_jaccard_per_class = val_metrics[6].cpu()
                        batch_sensitivity_per_class = val_metrics[7].cpu()
                        batch_specificity_per_class = val_metrics[8].cpu()

                        total_val_loss += batch_loss
                        total_val_dice_avg += batch_dice_avg
                        total_val_jaccard_avg += batch_jaccard_avg
                        total_val_sensitivity_avg += batch_sensitivity_avg
                        total_val_specificity_avg += batch_specificity_avg
                        total_val_dice_per_class += batch_dice_per_class
                        total_val_jaccard_per_class += batch_jaccard_per_class
                        total_val_sensitivity_per_class += batch_sensitivity_per_class
                        total_val_specificity_per_class += batch_specificity_per_class
                        
                    elif self.annotators_valid and not self.valid_ground_truth and len(data_batch) == 3:
                        images, masks, anns_onehot = data_batch
                        batch_loss = self.val_step(images, masks, anns_onehot)
                        total_val_loss += batch_loss
                        
                    elif not self.annotators_valid and self.valid_ground_truth and len(data_batch) == 2:
                        images, orig_mask = data_batch
                        val_metrics = self.val_step(images, None, None, orig_mask)
                        
                        batch_dice_avg = val_metrics[0].cpu()
                        batch_jaccard_avg = val_metrics[1].cpu()
                        batch_sensitivity_avg = val_metrics[2].cpu()
                        batch_specificity_avg = val_metrics[3].cpu()
                        batch_dice_per_class = val_metrics[4].cpu()
                        batch_jaccard_per_class = val_metrics[5].cpu()
                        batch_sensitivity_per_class = val_metrics[6].cpu()
                        batch_specificity_per_class = val_metrics[7].cpu()

                        total_val_dice_avg += batch_dice_avg
                        total_val_jaccard_avg += batch_jaccard_avg
                        total_val_sensitivity_avg += batch_sensitivity_avg
                        total_val_specificity_avg += batch_specificity_avg
                        total_val_dice_per_class += batch_dice_per_class
                        total_val_jaccard_per_class += batch_jaccard_per_class
                        total_val_sensitivity_per_class += batch_sensitivity_per_class
                        total_val_specificity_per_class += batch_specificity_per_class

                    num_val_batches += 1

                # Calculate validation averages
                if self.annotators_valid:
                    avg_val_loss = total_val_loss / num_val_batches
                else:
                    avg_val_loss = 0.0
                    
                if self.valid_ground_truth:
                    avg_val_dice = total_val_dice_avg / num_val_batches
                    avg_val_jaccard = total_val_jaccard_avg / num_val_batches
                    avg_val_sensitivity = total_val_sensitivity_avg / num_val_batches
                    avg_val_specificity = total_val_specificity_avg / num_val_batches
                    avg_val_dice_per_class = total_val_dice_per_class / num_val_batches
                    avg_val_jaccard_per_class = total_val_jaccard_per_class / num_val_batches
                    avg_val_sensitivity_per_class = total_val_sensitivity_per_class / num_val_batches
                    avg_val_specificity_per_class = total_val_specificity_per_class / num_val_batches
                else:
                    avg_val_dice = 0.0
                    avg_val_jaccard = 0.0
                    avg_val_sensitivity = 0.0
                    avg_val_specificity = 0.0
                    avg_val_dice_per_class = torch.zeros(num_classes)
                    avg_val_jaccard_per_class = torch.zeros(num_classes)
                    avg_val_sensitivity_per_class = torch.zeros(num_classes)
                    avg_val_specificity_per_class = torch.zeros(num_classes)

            # Generate visualizations every 5 epochs
            if epoch % 5 == 0 and len(data_batch) >= 3:
                if len(data_batch) == 4:
                    images, masks, anns_onehot, _ = data_batch
                else:
                    images, masks, anns_onehot = data_batch
                self.visualizations(images, masks, anns_onehot, epoch)

            # Save best model
            if self.valid_ground_truth and avg_val_dice > self.best_val_dice:
                self.best_val_dice = avg_val_dice
                torch.save(self.model.state_dict(), f'{self.models_dir}/best_model.pt')

            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)

            # Print and log metrics
            message = f"Train loss: {'zero' if avg_train_loss == 0 else f'{avg_train_loss:.5f}'} | Time: {elapsed_minutes}m {elapsed_seconds}s"
            self.train_loss[epoch] = avg_train_loss
            if self.wandb_monitoring:
                self.run.log({'Training Loss': avg_train_loss}, step=epoch)

            if self.valid_loader is not None and self.annotators_valid:
                message += f" | Val loss: {'zero' if avg_val_loss == 0 else f'{avg_val_loss:.5f}'}"
                self.val_loss[epoch] = avg_val_loss
                if self.wandb_monitoring:
                    self.run.log({'Validation Loss': avg_val_loss}, step=epoch)

            print(message)

            # Print training metrics if ground truth is available
            if self.train_ground_truth:
                train_metrics_avg = (
                    f"Train_DICE_avg: {'zero' if avg_train_dice == 0 else f'{avg_train_dice:.5f}'} | "
                    f"Train_Jaccard_avg: {'zero' if avg_train_jaccard == 0 else f'{avg_train_jaccard:.5f}'} | "
                    f"Train_Sensitivity_avg: {'zero' if avg_train_sensitivity == 0 else f'{avg_train_sensitivity:.5f}'} | "
                    f"Train_Specificity_avg: {'zero' if avg_train_specificity == 0 else f'{avg_train_specificity:.5f}'} "
                )
                print(train_metrics_avg)

                self.train_global_dice[epoch] = avg_train_dice
                self.train_global_jaccard[epoch] = avg_train_jaccard
                self.train_global_sensitivity[epoch] = avg_train_sensitivity
                self.train_global_specificity[epoch] = avg_train_specificity

                for c in range(num_classes):
                    train_metrics_class = (
                        f"Class {c} - Train_DICE: {'zero' if avg_train_dice_per_class[c] == 0 else f'{avg_train_dice_per_class[c]:.5f}'} | "
                        f"Train_Jaccard: {'zero' if avg_train_jaccard_per_class[c] == 0 else f'{avg_train_jaccard_per_class[c]:.5f}'} | "
                        f"Train_Sensitivity: {'zero' if avg_train_sensitivity_per_class[c] == 0 else f'{avg_train_sensitivity_per_class[c]:.5f}'} | "
                        f"Train_Specificity: {'zero' if avg_train_specificity_per_class[c] == 0 else f'{avg_train_specificity_per_class[c]:.5f}'} "
                    )
                    print(train_metrics_class)
                    self.train_per_class_dice[c][epoch] = avg_train_dice_per_class[c]
                    self.train_per_class_jaccard[c][epoch] = avg_train_jaccard_per_class[c]
                    self.train_per_class_sensitivity[c][epoch] = avg_train_sensitivity_per_class[c]
                    self.train_per_class_specificity[c][epoch] = avg_train_specificity_per_class[c]

                if self.wandb_monitoring:
                    self.run.log({
                        'Training DICE': avg_train_dice,
                        'Training Jaccard': avg_train_jaccard,
                        'Training Sensitivity': avg_train_sensitivity,
                        'Training Specificity': avg_train_specificity
                    }, step=epoch)

            # Print validation metrics if ground truth is available
            if self.valid_loader is not None and self.valid_ground_truth:
                val_metrics_avg = (
                    f"Val_DICE_avg: {'zero' if avg_val_dice == 0 else f'{avg_val_dice:.5f}'} | "
                    f"Val_Jaccard_avg: {'zero' if avg_val_jaccard == 0 else f'{avg_val_jaccard:.5f}'} | "
                    f"Val_Sensitivity_avg: {'zero' if avg_val_sensitivity == 0 else f'{avg_val_sensitivity:.5f}'} | "
                    f"Val_Specificity_avg: {'zero' if avg_val_specificity == 0 else f'{avg_val_specificity:.5f}'} "
                )
                print(val_metrics_avg)

                self.val_global_dice[epoch] = avg_val_dice
                self.val_global_jaccard[epoch] = avg_val_jaccard
                self.val_global_sensitivity[epoch] = avg_val_sensitivity
                self.val_global_specificity[epoch] = avg_val_specificity

                for c in range(num_classes):
                    val_metrics_class = (
                        f"Class {c} - Val_DICE: {'zero' if avg_val_dice_per_class[c] == 0 else f'{avg_val_dice_per_class[c]:.5f}'} | "
                        f"Val_Jaccard: {'zero' if avg_val_jaccard_per_class[c] == 0 else f'{avg_val_jaccard_per_class[c]:.5f}'} | "
                        f"Val_Sensitivity: {'zero' if avg_val_sensitivity_per_class[c] == 0 else f'{avg_val_sensitivity_per_class[c]:.5f}'} | "
                        f"Val_Specificity: {'zero' if avg_val_specificity_per_class[c] == 0 else f'{avg_val_specificity_per_class[c]:.5f}'} "
                    )
                    print(val_metrics_class)
                    self.val_per_class_dice[c][epoch] = avg_val_dice_per_class[c]
                    self.val_per_class_jaccard[c][epoch] = avg_val_jaccard_per_class[c]
                    self.val_per_class_sensitivity[c][epoch] = avg_val_sensitivity_per_class[c]
                    self.val_per_class_specificity[c][epoch] = avg_val_specificity_per_class[c]

                if self.wandb_monitoring:
                    self.run.log({
                        'Validation DICE': avg_val_dice,
                        'Validation Jaccard': avg_val_jaccard,
                        'Validation Sensitivity': avg_val_sensitivity,
                        'Validation Specificity': avg_val_specificity
                    }, step=epoch)

        # Cleanup and finalization
        gc.collect()

        if self.wandb_monitoring:
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file(f'{self.models_dir}/best_model.pt')
            self.run.log_artifact(artifact)
            self.run.finish()

        torch.save(self.model.state_dict(), f'{self.models_dir}/last_model.pt')
        torch.cuda.empty_cache()
        print("\nTraining complete!")

    def start(self) -> None:
        """Initialize and start the training process."""

        self.epochs = self.config.get('Epochs', 50)
        self.device = torch.device(self.config.get('Device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.amp_ = self.config.get('AMixPre', False)

        if self.amp_ and torch.cuda.is_available():
            self.scaler = GradScaler() if self.amp_ else None
            print("Automatic Mixed Precision (AMP) enabled.")

        self.print_device_info()

        self.run = None
        self.wandb_monitoring = self.config.get('Wandb monitoring', None)
        self.train_ground_truth = self.config.get('Ground truth train', False)
        self.valid_ground_truth = self.config.get('Ground truth valid', False)
        self.annotators_valid = self.config.get('Annotators valid', True)
        self.single_class_train = self.config.get('Single class train', None)
        self.single_class_valid = self.config.get('Single class valid', None)
        
        self.models_dir = './models'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.model = self.model_handling()
        self.model.to(self.device)
        self.loss_fn = self.loss_handling()

        self.train_phases = self.config.get('Train phases', False)

        self.wandb_logging()

        self.training()

        self.perform_across_epochs()