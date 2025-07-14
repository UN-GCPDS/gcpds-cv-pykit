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
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from ..models import UNet
from ..losses import (DICELoss, CrossEntropyLoss, FocalLoss, TverskyLoss)
from typing import Union, List, Tuple, Optional, Dict, Any


class SegmentationModel_Trainer:
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
                'Model', 'Image size', 'Number of classes',
                'Backbone', 'Activation function', 'Loss function'
            }
            wandb_config = {key: value for key, value in self.config.items() if key not in keys_to_include}
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

        model = self.config.get('Model', 'UNet')

        match model:

            case 'UNet':
                
                self.model = UNet(
                    in_channels=self.config.get('Input size', [3])[0],
                    out_channels=self.config.get('Number of classes', 1),
                    pretrained=self.config.get('Pretrained', True),
                    final_activation= self.config.get('Activation function', None),
                )
                return self.model

            case _:
                raise ValueError(f"Unknown model type: {model}")
            
    def loss_handling(self) -> nn.Module:

        loss_fn = self.config.get('Loss function', 'DICE')

        match loss_fn:

            case 'DICE':
                return DICELoss(
                    smooth=self.config.get('Smooth', 1.0),
                    reduction=self.config.get('Reduction', 'mean')
                )

            case 'CrossEntropy':
                return CrossEntropyLoss(reduction=self.config.get('Reduction', 'mean'))

            case 'Focal':
                return FocalLoss(
                    alpha=self.config.get('Alpha', 0.25),
                    gamma=self.config.get('Gamma', 2.0),
                    reduction=self.config.get('Reduction', 'mean')
                )
            
            case 'Tversky':
                return TverskyLoss(
                    alpha=self.config.get('Alpha', 0.5),
                    beta=self.config.get('Beta', 0.5),
                    smooth=self.config.get('Smooth', 1.0),
                    reduction=self.config.get('Reduction', 'mean')
                )

            case _:
                raise ValueError(f"Unknown loss function: {loss_fn}")


    def training_phases(self, phase: int) -> None:

        match phase:
            case 1:
                params = []
                # Freeze encoder parameters
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                # Enable decoder and segmentation_head parameters
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.segmentation_head.parameters():
                    param.requires_grad = True

                params.extend([
                    {'params': self.model.decoder.parameters(), 'lr': 1e-4},
                    {'params': self.model.segmentation_head.parameters(), 'lr': 1e-4}
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

                # Enable decoder and segmentation_head parameters
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.segmentation_head.parameters():
                    param.requires_grad = True

                params = [
                    {'params': self.model.decoder.parameters(), 'lr': 1e-4},
                    {'params': self.model.segmentation_head.parameters(), 'lr': 1e-4},
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

                # Enable decoder and segmentation_head parameters
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.segmentation_head.parameters():
                    param.requires_grad = True

                params = [
                    {'params': self.model.decoder.parameters(), 'lr': 1e-4},
                    {'params': self.model.segmentation_head.parameters(), 'lr': 1e-4},
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

                # Enable decoder and segmentation_head parameters
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.segmentation_head.parameters():
                    param.requires_grad = True

                params = [
                    {'params': self.model.decoder.parameters(), 'lr': 1e-4},
                    {'params': self.model.segmentation_head.parameters(), 'lr': 1e-4},
                    {'params': bn_params, 'lr': 1e-5},
                    {'params': layer3_params, 'lr': 1e-5},
                    {'params': layer4_params, 'lr': 1e-5}
                ]
                self.optimizer = optim.Adam(params)
                self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.94)

    def perform_across_epochs(self) -> None:

        os.makedirs("results", exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(range(self.epochs), self.train_loss, label='Training Loss', marker='o')
        if self.valid_loader is not None:
            plt.plot(range(self.epochs), self.val_loss, label='Validation Loss', marker='x')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/Loss.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(range(self.epochs), self.train_global_dice, label='Training Global DICE', marker='o')
        for i in range(self.config['Number of classes']):
            plt.plot(range(self.epochs), self.train_per_class_dice[i], label=f'Training Class {i} DICE', linestyle='--')
        if self.valid_loader is not None:
            plt.plot(range(self.epochs), self.val_global_dice, label='Validation Global DICE', marker='x')
            for i in range(self.config['Number of classes']):
                plt.plot(range(self.epochs), self.val_per_class_dice[i], label=f'Validation Class {i} DICE', linestyle=':')
        plt.title('DICE Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('DICE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/DICE.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(range(self.epochs), self.train_global_jaccard, label='Training Global Jaccard', marker='o')
        for i in range(self.config['Number of classes']):
            plt.plot(range(self.epochs), self.train_per_class_jaccard[i], label=f'Training Class {i} Jaccard', linestyle='--')
        if self.valid_loader is not None:
            plt.plot(range(self.epochs), self.val_global_jaccard, label='Validation Global Jaccard', marker='x')
            for i in range(self.config['Number of classes']):
                plt.plot(range(self.epochs), self.val_per_class_jaccard[i], label=f'Validation Class {i} Jaccard', linestyle=':')
        plt.title('Jaccard Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Jaccard')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/Jaccard.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(range(self.epochs), self.train_global_sensitivity, label='Training Global Sensitivity', marker='o')
        for i in range(self.config['Number of classes']):
            plt.plot(range(self.epochs), self.train_per_class_sensitivity[i], label=f'Training Class {i} Sensitivity', linestyle='--')
        if self.valid_loader is not None:
            plt.plot(range(self.epochs), self.val_global_sensitivity, label='Validation Global Sensitivity', marker='x')
            for i in range(self.config['Number of classes']):
                plt.plot(range(self.epochs), self.val_per_class_sensitivity[i], label=f'Validation Class {i} Sensitivity', linestyle=':')
        plt.title('Sensitivity Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Sensitivity')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/Sensitivity.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(range(self.epochs), self.train_global_specificity, label='Training Global Specificity', marker='o')
        for i in range(self.config['Number of classes']):
            plt.plot(range(self.epochs), self.train_per_class_specificity[i], label=f'Training Class {i} Specificity', linestyle='--') 
        if self.valid_loader is not None:
            plt.plot(range(self.epochs), self.val_global_specificity, label='Validation Global Specificity', marker='x')
            for i in range(self.config['Number of classes']):
                plt.plot(range(self.epochs), self.val_per_class_specificity[i], label=f'Validation Class {i} Specificity', linestyle=':')
        plt.title('Specificity Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Specificity')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/Specificity.png')
        plt.close()

    def visualizations(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        epoch: int
    ) -> None:

        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.eval()
        sample_n = random.randint(0, images.shape[0] - 1)
        sample = images[sample_n:sample_n + 1].to(self.device)
        mask_sample = masks[sample_n:sample_n + 1]
        with torch.no_grad():
            if self.amp_:
                with autocast(self.device.type):
                    prediction = self.model(sample)
            else:
                prediction = self.model(sample)
        if self.config.get('Loss function', None) == 'Focal':
            prediction = torch.sigmoid(prediction)
        sample_np = sample.cpu().numpy().transpose(0, 2, 3, 1)
        prediction_np = prediction.cpu().numpy()
        mask_sample_np = mask_sample.cpu().numpy()

        elements_display = min(prediction_np.shape[1] if prediction_np.shape[1]!= 1 else 2, 9)
        fig, axs = plt.subplots(3, elements_display, figsize=(16, 5))

        axs[0,0].imshow(sample_np[0])
        axs[0,0].set_title('Input image')

        for idx, element in enumerate(random.sample(range(self.config['Number of classes']), elements_display if self.config['Number of classes'] >= 2 else 1)):
            axs[1,idx].imshow(np.where(prediction_np[0, element] > 0.5, 1, 0), vmin=0.0, vmax=1.0)
            axs[1,idx].set_title(f'Pred class {element}')

            if isinstance(self.single_class_train, int):
                axs[2,0].imshow(mask_sample_np[0, 0], vmin=0.0, vmax=1.0)
                axs[2,0].set_title(f"GT for single class")
            else:
                axs[2,idx].imshow(mask_sample_np[0, element], vmin=0.0, vmax=1.0)
                axs[2,idx].set_title(f"GT for class {element}")

        [ax.axis('off') for row in axs for ax in row]

        plt.tight_layout()
        plt.show()
        plt.close()

        if self.wandb_monitoring:
            self.run.log({f"Predictions_Epoch_{epoch}": wandb.Image(fig)})

    def calculate_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        ignore_value: float = 0.6,
        threshold: float = 0.5,
        smooth: float = 1e-7
    ) -> Tuple[torch.Tensor, ...]:

        y_true = y_true.float()
        if self.config.get('Loss function', None) == 'Focal':
            y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.float()
        ignore_tensor = torch.tensor(ignore_value, device=y_true.device)
        mask = (y_true != ignore_tensor)
        valid_images = torch.any(mask, dim=(2, 3))
        mask = mask.float()
        valid_count = torch.sum(valid_images.float(), dim=0)
        y_true = (y_true > threshold).float()
        y_pred = (y_pred > threshold).float()

        intersection = torch.sum(y_true * y_pred * mask, dim=(2, 3))
        sum_true = torch.sum(y_true * mask, dim=(2, 3))
        sum_pred = torch.sum(y_pred * mask, dim=(2, 3))

        union_dice = sum_true + sum_pred
        dice_per_class = torch.zeros_like(valid_count)
        for c in range(valid_count.shape[0]):
            if valid_count[c] > 0:
                class_dice = (2.0 * intersection[:, c] + smooth) / (union_dice[:, c] + smooth)
                dice_per_class[c] = torch.sum(class_dice) / (valid_count[c] + smooth)

        union_jaccard = sum_true + sum_pred - intersection
        jaccard_per_class = torch.zeros_like(valid_count)
        for c in range(valid_count.shape[0]):
            if valid_count[c] > 0:
                class_jaccard = (intersection[:, c] + smooth) / (union_jaccard[:, c] + smooth)
                jaccard_per_class[c] = torch.sum(class_jaccard) / (valid_count[c] + smooth)

        true_positives = intersection
        actual_positives = sum_true
        sensitivity_per_class = torch.zeros_like(valid_count)
        for c in range(valid_count.shape[0]):
            if valid_count[c] > 0:
                class_sensitivity = true_positives[:, c] / (actual_positives[:, c] + smooth)
                sensitivity_per_class[c] = torch.sum(class_sensitivity) / (valid_count[c] + smooth)

        true_negatives = torch.sum((1 - y_true) * (1 - y_pred) * mask, dim=(2, 3))
        actual_negatives = torch.sum((1 - y_true) * mask, dim=(2, 3))
        specificity_per_class = torch.zeros_like(valid_count)
        for c in range(valid_count.shape[0]):
            if valid_count[c] > 0:
                class_specificity = true_negatives[:, c] / (actual_negatives[:, c] + smooth)
                specificity_per_class[c] = torch.sum(class_specificity) / (valid_count[c] + smooth)

        dice_per_class = torch.where(torch.isnan(dice_per_class), torch.tensor(0.0, device=dice_per_class.device), dice_per_class)
        jaccard_per_class = torch.where(torch.isnan(jaccard_per_class), torch.tensor(0.0, device=jaccard_per_class.device), jaccard_per_class)
        sensitivity_per_class = torch.where(torch.isnan(sensitivity_per_class), torch.tensor(0.0, device=sensitivity_per_class.device), sensitivity_per_class)
        specificity_per_class = torch.where(torch.isnan(specificity_per_class), torch.tensor(0.0, device=specificity_per_class.device), specificity_per_class)

        dice_avg = torch.mean(dice_per_class)
        jaccard_avg = torch.mean(jaccard_per_class)
        sensitivity_avg = torch.mean(sensitivity_per_class)
        specificity_avg = torch.mean(specificity_per_class)

        return (
            dice_avg, jaccard_avg, sensitivity_avg, specificity_avg,
            dice_per_class, jaccard_per_class, sensitivity_per_class, specificity_per_class
        )

    def train_step(
        self,
        images: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[float, ...]:

        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model or optimizer not initialized.")

        self.model.train()
        images = images.to(self.device)
        masks = masks.to(self.device)
        self.optimizer.zero_grad()

        if self.amp_:
            if self.scaler is None:
                raise RuntimeError("AMP is enabled but scaler is not initialized.")
            with autocast(self.device.type):
                y_pred = self.model(images)
                loss = self.loss_fn(y_pred,masks)
            if not torch.isnan(loss):
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            y_pred = self.model(images)
            loss = self.loss_fn(y_pred,masks)
            if not torch.isnan(loss):
                loss.backward()
                self.optimizer.step()

        if isinstance(self.single_class_train, int):
            selected_pred = y_pred[:, self.single_class_train:self.single_class_train + 1]
            metrics = self.calculate_metrics(masks, selected_pred)
        else:
            selected_pred = y_pred[:, :self.config['Number of classes']]
            metrics = self.calculate_metrics(masks, selected_pred)

        return (loss.item(), *metrics)

    def val_step(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[float, ...]:

        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.eval()
        images = images.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)

        with torch.no_grad():
            if self.amp_:
                with autocast(self.device.type):
                    y_pred = self.model(images)
                    loss = self.loss_fn(y_pred,masks)
                    loss = loss.item() if not torch.isnan(loss) else 0.0
            else:
                y_pred = self.model(images)
                loss = self.loss_fn(y_pred,masks)
                loss = loss.item() if not torch.isnan(loss) else 0.0

            if isinstance(self.single_class_valid, int):
                selected_pred = y_pred[:, self.single_class_valid:self.single_class_valid + 1]
                metrics = self.calculate_metrics(masks, selected_pred)
            else:
                selected_pred = y_pred[:, :self.config['Number of classes']]
                metrics = self.calculate_metrics(masks, selected_pred)

            return (loss, *metrics)

    def training(self) -> None:

        self.best_train_dice = 0.0
        self.best_train_loss = float('inf')
        self.best_val_dice = 0.0
        self.best_val_loss = float('inf')

        self.train_loss = np.zeros(self.epochs)
        self.val_loss = np.zeros(self.epochs)

        self.train_global_dice = np.zeros(self.epochs)
        self.train_per_class_dice = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]
        self.train_global_jaccard = np.zeros(self.epochs)
        self.train_per_class_jaccard = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]
        self.train_global_sensitivity = np.zeros(self.epochs)
        self.train_per_class_sensitivity = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]
        self.train_global_specificity = np.zeros(self.epochs)
        self.train_per_class_specificity = [np.zeros(self.epochs) for _ in range(self.config['Number of classes'])]

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
                images, masks = data_batch
                train_metrics = self.train_step(images, masks)

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

                num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches
            avg_train_dice = total_train_dice_avg / num_train_batches
            avg_train_jaccard = total_train_jaccard_avg / num_train_batches
            avg_train_sensitivity = total_train_sensitivity_avg / num_train_batches
            avg_train_specificity = total_train_specificity_avg / num_train_batches
            avg_train_dice_per_class = total_train_dice_per_class / num_train_batches
            avg_train_jaccard_per_class = total_train_jaccard_per_class / num_train_batches
            avg_train_sensitivity_per_class = total_train_sensitivity_per_class / num_train_batches
            avg_train_specificity_per_class = total_train_specificity_per_class / num_train_batches

            if epoch > 30:
                self.scheduler.step()

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
                    images, masks = data_batch
                    val_metrics = self.val_step(images, masks)

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

                    num_val_batches += 1

                avg_val_loss = total_val_loss / num_val_batches
                avg_val_dice = total_val_dice_avg / num_val_batches
                avg_val_jaccard = total_val_jaccard_avg / num_val_batches
                avg_val_sensitivity = total_val_sensitivity_avg / num_val_batches
                avg_val_specificity = total_val_specificity_avg / num_val_batches
                avg_val_dice_per_class = total_val_dice_per_class / num_val_batches
                avg_val_jaccard_per_class = total_val_jaccard_per_class / num_val_batches
                avg_val_sensitivity_per_class = total_val_sensitivity_per_class / num_val_batches
                avg_val_specificity_per_class = total_val_specificity_per_class / num_val_batches
            else:
                avg_val_loss = 0.0
                avg_val_dice = 0.0
                avg_val_jaccard = 0.0
                avg_val_sensitivity = 0.0
                avg_val_specificity = 0.0
                avg_val_dice_per_class = torch.zeros(num_classes)
                avg_val_jaccard_per_class = torch.zeros(num_classes)
                avg_val_sensitivity_per_class = torch.zeros(num_classes)
                avg_val_specificity_per_class = torch.zeros(num_classes)

            if epoch % 5 == 0:
                self.visualizations(images, masks, epoch)

            if avg_val_dice > self.best_val_dice:
                self.best_val_dice = avg_val_dice
                torch.save(self.model.state_dict(), f'{self.models_dir}/best_model.pt')

            elapsed_time = time.time() - self.start_time
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)

            message = f"Train loss: {'zero' if avg_train_loss == 0 else f'{avg_train_loss:.5f}'} | Time: {elapsed_minutes}m {elapsed_seconds}s"
            self.train_loss[epoch] = avg_train_loss
            if self.wandb_monitoring:
                self.run.log({'Training Loss': avg_train_loss}, step=epoch)

            if self.valid_loader is not None:
                message += f" | Val loss: {'zero' if avg_val_loss == 0 else f'{avg_val_loss:.5f}'}"
                self.val_loss[epoch] = avg_val_loss
                if self.wandb_monitoring:
                    self.run.log({'Validation Loss': avg_val_loss}, step=epoch)

            print(message)

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
            
            if self.valid_loader is not None:

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

        self.epochs = self.config.get('Epochs',50)
        self.device = torch.device(self.config.get('Device', 'cpu'))
        self.amp_ = self.config.get('AMixPre', False)

        if self.amp_ and torch.cuda.is_available():
            self.scaler = GradScaler() if self.amp_ else None
            print("Automatic Mixed Precision (AMP) enabled.")

        self.print_device_info()

        self.run = None
        self.wandb_monitoring = self.config.get('Wandb monitoring', None)
        self.single_class_train = self.config.get('Single class train', None)
        self.single_class_valid = self.config.get('Single class valid', None)
        
        self.models_dir = './models'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.model = self.model_handling()
        self.model.to(self.device)
        self.loss_fn = self.loss_handling()

        self.wandb_logging()

        self.training()

        self.perform_across_epochs()