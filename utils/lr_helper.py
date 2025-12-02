import torch

def get_scheduler(optimizer, config):
    scheduler_type = config.type
    
    if scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **config.kwargs)
    elif scheduler_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config.kwargs)
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config.kwargs)
    elif scheduler_type == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **config.kwargs)
    elif scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.kwargs)
    elif scheduler_type == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.kwargs)
    elif scheduler_type == "LambdaLR":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, **config.kwargs)
    else:
        raise NotImplementedError(f"Learning rate scheduler '{scheduler_type}' is not implemented")
