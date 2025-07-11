import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy loss with logits for binary and multilabel segmentation/classification.

    This class expects:
    - logits: Tensor of shape (N, C, H, W), raw model outputs (no sigmoid).
    - labels: Tensor of shape (N, C, H, W), ground truth with values 0 or 1 (multi-hot).

    Each class/channel is treated independently, allowing multiple classes per pixel.

    Args:
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        torch.Tensor: Loss tensor. If reduction is 'none', same shape as inputs; else scalar.
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Model outputs of shape (N, C, H, W), raw logits.
            labels (torch.Tensor): Ground truth of shape (N, C, H, W), values 0 or 1.

        Returns:
            torch.Tensor: Scalar loss value (or unreduced loss if reduction='none').
        """
        labels = labels.float()
        return self.criterion(logits, labels)