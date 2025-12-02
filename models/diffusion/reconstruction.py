import torch
import numpy as np
from typing import Any, Optional


class Reconstruction:
    """
        DDIM sampling with conditioning
        - unet: unet denoising
        - trajectory_steps: total number of diffusion forward
        - test_trajectory_steps: number of step for reconstruction
        - skip: step size for sampling sequence
        - eta: DDIM eta parameter (0.0 for DDIM, 1.0 for DDPM)
        - beta_start: start beta value for noise schedule
        - beta_end: Ending beta value for noise schedule
    """
    def __init__(
        self,
        unet,
        trajectory_steps,
        test_trajectory_steps,
        skip,
        eta,
        beta_start,
        beta_end,
        device = None    
    ):
        assert unet is not None, "unet is not None"
        self.unet = unet
        self.trajectory_steps = trajectory_steps
        self.test_trajectory_steps = test_trajectory_steps
        self.skip = skip
        self.eta = eta
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
    
    def _compute_alpha(self, t, device):
        # linear beta scheduler
        betas = np.linspace(self.beta_start, self.beta_end, self.trajectory_steps, dtype=np.float64)
        betas = torch.tensor(betas, dtype=torch.float32, device=device)
        beta = torch.cat([torch.zeros(1, device=device), betas], dim=0) # [b1, b2, ...] -> # [0, b1, b2]
        alphas = (1 - beta).cumprod(dim=0) 
        a = alphas.index_select(0, t+1).view(-1, 1, 1, 1)
    
    def __call__(self, x, y0, w) -> Any:
        """
        - x: input image [B, C, H, W] - the image to reconstruct
        - y0: conditioning image [B, C, H, W]
        - w: weight for conditioning (DDAD mvtec = 2)
        return: 
            List of reconstructed images at each step, or final image if single step
        """
        device = x.device if self.device is None else self.device
        x = x.to(device)
        if y0 is not None:
            y0 = y0.to(device)
        
        num_steps = self.test_trajectory_steps
        # noise image
        at = self._compute_alpha(torch.tensor([num_steps], device=device, dtype=torch.long), device)
        xt = at.sqrt() * x + (1 - at).sqrt() * torch.randn_like(x).to(device)

        # sampling sequence
        seq = list(range(0, num_steps, self.skip))
        seq_next = [-1] + list(seq[:-1])

        with torch.no_grad():
            n = x.size(0)
            xs = [xt]

            for i,j in zip(reversed(seq), reversed(seq_next)):
                t = torch.ones(n, device=device, dtype=torch.long) * i
                next_t = torch.ones(n, device=device, dtype=torch.long) * j

                at = self._compute_alpha(t, device)
                at_next = self._compute_alpha(next_t, device) if j >= 0 else torch.ones_like(at)
                xt = xs[-1].to(device)

                # predict noise
                et = self.unet(xt, t)
                # Conditional image
                yt = at.sqrt() * y0 + (1 - at).sqrt() * et
                # Adjusted noise
                et_hat = et - (1 - at).sqrt() * w * (yt - xt)
                # predict x0
                x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()

                c1 = self.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat

                xs.append(xt_next)
        
        return xs


