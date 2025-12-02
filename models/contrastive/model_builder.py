import importlib
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from models.backbones import backbone_info

def build_module(
    module_config: Dict[str, Any],
    built_modules: Optional[Dict[str, nn.Module]] = None
) -> nn.Module:
    
    if built_modules is None:
        built_modules = {}
    
    module_type = module_config['type']
    kwargs = module_config.get('kwargs', {}).copy()
    
    for key, value in kwargs.items():
        if isinstance(value, str) and value in built_modules:
            kwargs[key] = built_modules[value]

    module_path, class_name = module_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


class ContrastiveModelBuilder(nn.Module):

    def __init__(self, cfg, h_dim=256) -> None:
        super(ContrastiveModelBuilder, self).__init__()
        self.cfg = cfg
        self.cfg = cfg
        self.h_dim = h_dim
        net_config = cfg.get('contrastive_net', [])
        built_modules = {}
        
        # build all modules
        for module_config in net_config:
            module_name = module_config['name']
            
            config_copy = module_config.copy()
            if module_name == 'backbone':
                kwargs = config_copy.get('kwargs', {}).copy()
                outlayers = kwargs.get('outlayers')
            
                # backbone from module_type
                module_type = config_copy.get('type', '')
                backbone_name = 'efficientnet_b4'

                if backbone_name and backbone_name in backbone_info:
                    info = backbone_info[backbone_name]
                    outblocks = [info['blocks'][i-1] for i in outlayers if i <= len(info['blocks'])]
                    outstrides = [info['strides'][i-1] for i in outlayers if i <= len(info['strides'])]
                    kwargs['outblocks'] = outblocks
                    kwargs['outstrides'] = outstrides
                    kwargs.pop('outlayers', None)
                    self._backbone_info = {
                        'outlayers': outlayers, 
                        'outblocks': outblocks,
                        'outstrides': outstrides,
                        'planes': [info['planes'][i-1] for i in outlayers if i <= len(info['planes'])]
                    }

                config_copy['kwargs'] = kwargs
                # handle backbone as function, not class
                if 'models.backbones.' in module_type:
                    try:
                        # Backbone is a function, call it directly
                        module_path, func_name = module_type.rsplit('.', 1)
                        backbone_module = importlib.import_module(module_path)
                        backbone_func = getattr(backbone_module, func_name)
                        # Call the function with kwargs
                        module = backbone_func(**kwargs)
                        self.add_module(module_name, module)
                        built_modules[module_name] = module
                        continue  
                    except Exception as e:
                        raise RuntimeError(f"Failed to build backbone module '{module_name}': {str(e)}")
            
            elif module_name == 'neck':
                if 'backbone' not in built_modules:
                    raise ValueError('Backbone must init before neck')
            
                kwargs = config_copy.get('kwargs', {}).copy()
                # get backbone output info
                backbone_module = built_modules['backbone']
                if hasattr(backbone_module, 'get_outplanes'):
                    inplanes = backbone_module.get_outplanes()
                elif hasattr(self, '_backbone_info'):
                    inplanes = self._backbone_info['planes']
                else:
                    raise ValueError('cannot backbone outplanes')
                
                if hasattr(backbone_module, 'get_outstrides'):
                    instrides = backbone_module.get_outstrides()
                elif hasattr(self, '_backbone_info'):
                    instrides = self._backbone_info['outstrides']
                else:
                    raise ValueError('cannot backbone outstrides')
                
                outstrides = kwargs.get('outstrides', [16])
                # sum planes for mncf
                outplanes = [sum(inplanes)]
                kwargs['inplanes'] = inplanes
                kwargs['instrides'] = instrides
                kwargs['outplanes'] = outplanes

                config_copy['kwargs'] = kwargs

            elif module_name == 'arl':
                kwargs = config_copy.get('kwargs', {}).copy()
                # gumbel
                gumbel_aug = built_modules.get('gumbel_aug', None)
                if gumbel_aug is not None:
                    kwargs['gumbel_aug'] = gumbel_aug
                
                # backbone and neck
                if 'backbone' not in built_modules:
                    raise ValueError('backbone must init before ARL')
                elif 'neck' not in built_modules:
                    raise ValueError('neck must init before ARL')

                backbone = built_modules['backbone']
                neck = built_modules['neck']

                kwargs['backbone'] = backbone
                kwargs['neck'] = neck
                kwargs['cfg'] = cfg
                kwargs['h_dim'] = self.h_dim

                if hasattr(self, '_backbone_info'):
                    kwargs['outlayers'] = self._backbone_info['outlayers']
                else: 
                    kwargs['outlayers'] = [1, 2, 3, 4]
                
                config_copy['kwargs'] = kwargs

            try:
                module = build_module(config_copy, built_modules)
                self.add_module(module_name, module)
                built_modules[module_name] = module
            except Exception as e:
                raise RuntimeError(f"Failed to build module '{module_name}': {str(e)}")
            
        if 'arl' not in built_modules:
            raise ValueError(
                "ARL module must be defined in contrastive_net config"
            )
    

    def forward(self, x: torch.Tensor):
        #print(f"Type of self.arl: {type(self.arl)}")  # Debug
        #print(f"Is nn.Module: {isinstance(self.arl, nn.Module)}")  # Debug
        embeddings = self.arl(x)
        return embeddings


    def cuda(self):
        self.device = torch.device("cuda")
        return super(ContrastiveModelBuilder, self).cuda()
    
    
    def cpu(self):
        self.device = torch.device("cpu")
        return super(ContrastiveModelBuilder, self).cpu()
    

    def train(self, mode=True):
        self.training = mode
        if self.arl is not None:
            self.arl.train(mode)
        else:
            for module in self.children():
                module.train(mode)
        return self
                

            

