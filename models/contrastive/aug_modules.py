import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import torch.nn.functional as F

"""
- GumbelSoftmax augmentation
"""
class GumbelAugmentation(nn.Module):
      def __init__(
            self,
            aug_linear,
            aug_cnn,
            tau, # float
            hard # bool
      ):
            super().__init__()
            self.branches = nn.ModuleList([aug_linear, aug_cnn])
            self.num_branches = len(self.branches)
            self.tau = tau
            self.hard = hard
            self.logits = nn.Parameter(
                  torch.zeros(self.num_branches, dtype=torch.float32)
            )  # shape [num_branches]
            
      def forward(self, x):
            B = x.shape[0]
            # augment linear and cnn
            aug_outs = [
                  aug_module(x) for aug_module in self.branches
            ]  # list of [B, C, H, W]
            
            ref_shape = aug_outs[0].shape 
            for i, out in enumerate(aug_outs):
                  if out.shape != ref_shape:
                        raise RuntimeError(
                              f"All branches must return the same shape. "
                              f"Branch 0: {ref_shape}, branch {i}: {out.shape}"
                        )
            
            # stack [B, num_branches, C, H, W]
            stack = torch.stack(aug_outs, dim=1)
            
            # tạo trọng số Gumbel-Softmax, [B, num_branches]
            logits = self.logits.unsqueeze(0).expand(B, -1)
            w = F.gumbel_softmax(
                  logits, tau=self.tau, hard=self.hard, dim=-1
            )  # [B, num_branches]

            weights = w[..., None, None, None] # [B, nb, 1, 1, 1]
            y = (weights * stack).sum(dim=1) # [B, C, H, W]

            return y # mixed augmentation output
            


"""
- Linear Augmentation
"""
class LinearAugmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # linear 1x1 conv
        self.mix = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self._init_mix_as_identity()
        # y = alpha * x + beta
        self.alpha = nn.Parameter(torch.ones(self.out_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(self.out_channels, 1, 1))

    def _init_mix_as_identity(self):
        with torch.no_grad():
            self.mix.weight.zero_()
            for c in range(self.out_channels):
                self.mix.weight[c, c, 0, 0] = 1.0
            if self.mix.bias is not None:
                self.mix.bias.zero_()

    def forward(self, x: torch.Tensor):
        """
        x: [B, in_channels, H, W]
        return: [B, out_channels, H, W]
        """
        y = self.mix(x)
        y = y * self.alpha + self.beta
        return y


"""
- CNN Augmentation
"""
class CNNAugmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        n_blocks: int,
        num_groups: int,
        bias: bool = True
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_groups = num_groups
        self.bias = bias

        self.blocks = nn.ModuleList(
            Block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                num_groups=num_groups,
                padding=padding,
                bias=bias
            ) 
            for _ in range(self.n_blocks)
        )
        # gate
        self.gates = nn.Parameter(torch.zeros(self.n_blocks))
    
    def forward(self, x):
        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        
        y = x
        for i, block in enumerate(self.blocks):
            z = block(y)
            gate = torch.sigmoid(self.gates[i]) # [0,1]
            y = y + gate * z
        
        if squeeze:
            y = y.squeeze(0)
        return y


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int, # 3
        num_groups:int, 
        padding: int,
        bias: bool = True, # True
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=padding, 
            bias=bias
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activation = nn.SiLU()
        

    def forward(self, x):
        y = self.conv2d(x)
        y = self.norm(y)
        y = self.activation(y)
        return y



