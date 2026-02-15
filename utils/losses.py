import torch
import torch.nn as nn

class QLIKELoss(nn.Module):
    def __init__(self, epsilon=1e-5, reduction='mean'):
        super(QLIKELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # 确保预测值和真实值都为正，以保证数值稳定性
        y_pred = torch.clamp(y_pred, min=self.epsilon)
        y_true = torch.clamp(y_true, min=self.epsilon)
        
        # 计算每个元素的QLIKE损失
        ratio = y_true / y_pred
        loss = ratio - torch.log(ratio) - 1
        
        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"不支持的reduction类型: {self.reduction}")