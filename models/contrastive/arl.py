import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class ARL(nn.Module):
      def __init__(
            self,
            cfg,
            h_dim,
            gumbel_aug = None,
            backbone = "efficientnet-b4",
            outlayers = [1,2,3,4],
            neck = None,
            # is_sepreted_mlp = False
      ):
            super().__init__()
            self.cfg = cfg
            self.h_dim = h_dim
            self.backbone = backbone
            self.gumbel_aug = gumbel_aug
            self.neck = neck
            # Get pixel_mean and pixel_std from dataset config
            dataset_cfg = cfg.get("dataset", {})
            pixel_mean = dataset_cfg.get("pixel_mean")
            pixel_std = dataset_cfg.get("pixel_std")
            self.normalize_fn = transforms.Normalize(
                  mean=pixel_mean, 
                  std=pixel_std
            )
            
            # Calculate feature_dim from neck output
            if neck is not None:
                  outplanes = neck.get_outplanes()
                  if isinstance(outplanes, list) and len(outplanes) > 0:
                        self.feature_dim = outplanes[0]
                  else:
                        raise ValueError("Cannot determine feature_dim from neck")
            else:
                  raise ValueError("Neck must be provided to determine feature_dim")
            
            # mlp
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                  nn.Linear(self.feature_dim, self.feature_dim),
                  nn.ReLU(),
                  nn.Linear(self.feature_dim, self.h_dim)
            )
      
      def forward(self, x):
            """
            Forward pass through ARL.
            Args:
                  x: Original image tensor [B, C, H, W]
                  
            Returns:
                  embeddings: Concatenated embeddings [B, h_dim * 2]
            """
            # Normalize input image first
            normed_x = self.normalize_fn(x)
            
            # gumbel augment
            aug_image = self.gumbel_aug(x) if self.gumbel_aug is not None else x

            # normalize aug_image
            normed_aug_image = self.normalize_fn(aug_image)
            embeddings_list = []

            # backbone
            features_aug = self.backbone({"image": normed_aug_image})
            features_aug_multiscale = self.neck(features_aug)

            features_origin = self.backbone({"image": normed_x})
            features_origin_multiscale = self.neck(features_origin)

            # Neck returns {"feature_align": tensor, "outplane": list}
            # feature_align is already concatenated [B, feature_dim, H, W]
            feature_aug_align = features_aug_multiscale["feature_align"]  # [B, feature_dim, H, W]
            feature_origin_align = features_origin_multiscale["feature_align"]  # [B, feature_dim, H, W]
            
            # pool and flatten
            pooled_aug = self.pool(feature_aug_align)  # [B, feature_dim, 1, 1]
            pooled_aug = pooled_aug.view(pooled_aug.size(0), -1)  # [B, feature_dim]
            embeddings_aug = self.mlp(pooled_aug)  # [B, h_dim]
            embeddings_aug = F.normalize(embeddings_aug, dim=1)  # L2 normalize
            embeddings_list.append(embeddings_aug)
            
            pooled_origin = self.pool(feature_origin_align)  # [B, feature_dim, 1, 1]
            pooled_origin = pooled_origin.view(pooled_origin.size(0), -1)  # [B, feature_dim]
            embeddings_origin = self.mlp(pooled_origin)  # [B, h_dim]
            embeddings_origin = F.normalize(embeddings_origin, dim=1)  # L2 normalize
            embeddings_list.append(embeddings_origin)

            # Concatenate embeddings [B, h_dim * num_images]
            embeddings = torch.cat(embeddings_list, dim=1)  # [B, h_dim * num_images]
            return embeddings