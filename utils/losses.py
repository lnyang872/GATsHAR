import torch
import torch.nn as nn

class QLIKELoss(nn.Module):
    def __init__(self, epsilon=1e-5, reduction='mean'):
        super(QLIKELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Ensure both predicted and actual values are positive to guarantee numerical stability.
        y_pred = torch.clamp(y_pred, min=self.epsilon)
        y_true = torch.clamp(y_true, min=self.epsilon)
        
        # Compute the QLIKE loss for each element
        ratio = y_true / y_pred
        loss = ratio - torch.log(ratio) - 1
        
        # Aggregate losses based on the reduction parameter
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction types: {self.reduction}")
