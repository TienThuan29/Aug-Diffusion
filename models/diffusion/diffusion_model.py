import torch
import torch.nn as nn
import math
from typing import Any, Optional


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    f_t = torch.cos(((t / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, min=1e-4, max=0.999)


class DiffusionModel(nn.Module):

    def __init__(
        self, 
        unet: nn.Module, 
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        reconstruction_params: Any = None,
        reconstruction: Any = None,
    ):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.reconstruction_params = reconstruction_params
        # Reconstruction instance (built by model_builder)
        self.reconstruction = reconstruction
        
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
    

    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise
    

    def forward(self, x_0):
        batch_size = x_0.shape[0]   
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device)
        x_t, noise = self.add_noise(x_0, t)
        noise_pred = self.unet(x_t, t)

        return {
            "noise_pred": noise_pred,
            "noise": noise,
            "x_t": x_t,
            "t": t
        }
    
    def reconstruct(self, x_0, y0: Optional[torch.Tensor] = None, w: float = 1.0, num_steps: Optional[int] = None):
        if self.reconstruction is None:
            raise ValueError("reconstruction class is not initialized")
        
        xs = self.reconstruction(x_0, y0=y0, w=w, num_steps=num_steps)
        return xs[-1]
    