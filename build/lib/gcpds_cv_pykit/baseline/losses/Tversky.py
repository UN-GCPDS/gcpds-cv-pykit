import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary and multilabel segmentation.

    Args:
        alpha (float): Weight for false positives (default: 0.5).
        beta (float): Weight for false negatives (default: 0.5).
        smooth (float): Smoothing constant to avoid division by zero (default: 1.0).
        reduction (str): 'mean' | 'sum' | 'none' (default: 'mean').

    Inputs:
        preds (torch.Tensor): Predicted probabilities, shape (N, C, H, W).
        targets (torch.Tensor): Ground truth, same shape, values 0 or 1.

    Returns:
        torch.Tensor: Tversky loss (scalar if reduction is 'mean' or 'sum').
    """
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds (torch.Tensor): Predicted probabilities, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth, same shape, values 0 or 1.

        Returns:
            torch.Tensor: Tversky loss.
        """
        preds = preds.float()
        targets = targets.float()
        N, C = preds.shape[:2]
        preds = preds.view(N, C, -1)
        targets = targets.view(N, C, -1)

        # True Positives, False Positives & False Negatives
        tp = (preds * targets).sum(-1)
        fp = (preds * (1 - targets)).sum(-1)
        fn = ((1 - preds) * targets).sum(-1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = 1 - tversky

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

# Example usage:
if __name__ == "__main__":
    loss_fn = TverskyLoss(alpha=0.7, beta=0.3, reduction='mean')
    preds = torch.sigmoid(torch.randn(2, 3, 128, 128))  # probabilities
    targets = torch.randint(0, 2, (2, 3, 128, 128)).float()
    loss = loss_fn(preds, targets)
    print(loss)