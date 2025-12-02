import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


class NTXentLoss(nn.Module):
    def __init__(self, temp=0.07, normalize=True, reduction="mean", **kwargs) -> None:
        super(NTXentLoss, self).__init__()
        self.normalize = normalize
        self.reduction = reduction
        self.ntxent_loss = losses.NTXentLoss(temperature=temp, **kwargs)

    def forward(self, z: torch.Tensor, pos_index: torch.LongTensor | None = None):
        """ Reshape embedding trước khi vào loss này """
        B = z.shape[0] // 2
        if self.normalize:
            z = F.normalize(z, p=2, dim=1)
        labels = torch.arange(B, device=z.device).repeat(2)
        loss = self.ntxent_loss(z, labels)
        
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum() if loss.dim() > 0 else loss
        else: 
            return loss.mean() if loss.dim() > 0 else loss


class MSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        predicted = input["recon"]
        target = input["origin"]
        return self.criterion_mse(predicted, target)


class DiffusionLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight
    
    def forward(self, input):
        predicted_noise = input["noise_pred"]
        target_noise = input["noise"]
        return self.criterion_mse(predicted_noise, target_noise)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict