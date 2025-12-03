import importlib
import torch
import torch.nn as nn
from torch.utils.data import dataset
import torchvision.transforms as transforms
from typing import Dict, Any, Optional
from .reconstruction import Reconstruction


def build_module(
    module_config: Dict[str, Any], 
    built_modules: Optional[Dict[str, nn.Module]] = None
):
    if built_modules is None:
        built_modules = {}
    
    module_type = module_config['type']
    kwargs = module_config.get('kwargs', {}).copy()
    # build module
    for key, value in kwargs.items():
        if isinstance(value, dict) and 'type' in value:
            nested_kwargs = value.get("kwargs", {}).copy()
            nested_module_type = value['type']
            nested_module_path, nested_class_name = nested_module_type.rsplit('.', 1)
            nested_module = importlib.import_module(nested_module_path)
            nested_cls = getattr(nested_module, nested_class_name)
            kwargs[key] = nested_cls(**nested_kwargs)
        elif isinstance(value, str) and value in built_modules:
            kwargs[key] = built_modules[value]
    
    module_path, class_name = module_type.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    return cls(**kwargs)


class DiffusionModelBuilder(nn.Module):

    def __init__(
        self,
        cfg,
        contrastive_model
    ):
        super(DiffusionModelBuilder, self).__init__()
        self.cfg = cfg

        dataset_cfg = cfg.get('dataset')
        # build gumbel_aug
        contrastive_net_config = cfg.get('contrastive_net', [])
        built_modules = {}

        for module_config in contrastive_net_config:
            module_name = module_config['name']
            # build aug modules
            if module_name in ['linear_aug', 'cnn_aug', 'gumbel_aug']:
                try:
                    # Add input_normalized=True for augmentation modules to work with normalized images
                    config_copy = module_config.copy()
                    config_copy.setdefault('kwargs', {})
                    module = build_module(config_copy, built_modules)
                    self.add_module(module_name, module)
                    built_modules[module_name] = module
                except Exception as e:
                    raise RuntimeError(f"Failed to build module '{module_name}': {str(e)}")
            
        if 'gumbel_aug' not in built_modules:
            raise ValueError("gumbel_aug module not found in contrastive_net config")
        
        self.gumbel_aug = built_modules['gumbel_aug']
        # load contrastive model
        self.gumbel_aug.load_state_dict(contrastive_model.gumbel_aug.state_dict())
        print("loaded pretrained gumbel_aug")
        # frozen gumbel
        for p in self.gumbel_aug.parameters():
            p.requires_grad = False

        # build diffusion model
        # diffusion_net_config = cfg.get('diffusion_net', [])
        # diffusion_config = module_config.copy()
        
        # if diffusion_config is None:
        #     raise ValueError("diffusion module not found in diffusion_net config")
        diffusion_net_config = cfg.get('diffusion_net', [])
        diffusion_config = None
        for m in diffusion_net_config:
            if m.get('name') == 'diffusion':
                diffusion_config = m.copy()
                break
        if diffusion_config is None:
            raise ValueError("diffusion module not found in diffusion_net config")
        
        # extract reconstruction_params
        diffusion_kwargs = diffusion_config.get('kwargs', {}).copy()
        if 'reconstruction_params' in diffusion_kwargs:
            recon_config = diffusion_kwargs.pop('reconstruction_params')
            reconstruction_params = recon_config.get('kwargs', {})
        
        diffusion_config['kwargs'] = diffusion_kwargs

        # build diffusion model
        self.diffusion = build_module(diffusion_config, {})
        # reconstruction param
        traj_steps = reconstruction_params.get('trajectory_steps')
        test_traj_steps = reconstruction_params.get('test_trajectory_steps')
        skip = reconstruction_params.get('skip')
        eta = reconstruction_params.get('eta')
        beta_start = diffusion_kwargs.get('beta_start')
        beta_end = diffusion_kwargs.get('beta_end')
        # build reconstruction
        reconstruction = Reconstruction(
            unet=self.diffusion.unet,
            trajectory_steps=traj_steps,
            test_trajectory_steps=test_traj_steps,
            skip=skip,
            eta=eta,
            beta_start=beta_start,
            beta_end=beta_end
        )

        self.diffusion.reconstruction = reconstruction
        self.diffusion.reconstruction_params = reconstruction_params
        self.add_module('diffusion', self.diffusion)
    

    def forward(self, x):
        self.gumbel_aug.eval()
        x_aug = self.gumbel_aug(x)
        diffusion_output = self.diffusion(x_aug)
        return diffusion_output
    

    def load_pretrained_gumbel_aug(self, checkpoint_path: str, strict: bool = True):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        gumbel_state_dict = {}
        for key, value in state_dict.items():
            if 'gumbel_aug' in key:
                # Remove 'gumbel_aug.' prefix if present
                new_key = key.replace('gumbel_aug.', '')
                gumbel_state_dict[new_key] = value
        
        if not gumbel_state_dict:
            raise ValueError(f"No gumbel_aug parameters found in checkpoint: {checkpoint_path}")
        
        self.gumbel_aug.load_state_dict(gumbel_state_dict, strict=strict)
        print("loaded pretrained gumbel_aug")
        # frozen gumbel
        for p in self.gumbel_aug.parameters():
            p.requires_grad = False
    

    def cuda(self):
        self.device = torch.device("cuda")
        return super(DiffusionModelBuilder, self).cuda()
    

    def cpu(self):
        self.device = torch.device("cpu")
        return super(DiffusionModelBuilder, self).cpu()
    

    def train(self, mode=True):
        self.training = mode
        if hasattr(self, 'diffusion'):
            self.diffusion.train(mode)
        return self

            
