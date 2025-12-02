from .dataset import build_custom_dataset

def build(cfg, training, class_name=None):
      if training:
            cfg.update(cfg.get("train", {}))
      else:
            cfg.update(cfg.get("test", {}))

      dataset = cfg["type"]
      if dataset == "custom":
            dataloader = build_custom_dataset(cfg, training, class_name)
      else:
            raise NotImplementedError(f"dataset {dataset} is not supported")
      
      return dataloader      
      

def build_dataloader(cfg_dataset, class_name=None):
      train_loader = None
      if cfg_dataset.get("train", None):
            train_loader = build(cfg_dataset, training=True, class_name=class_name)
            
      test_loader = None
      if cfg_dataset.get("test", None):
            test_loader = build(cfg_dataset, training=False, class_name=class_name)
      
      print("build dataset done")
      return train_loader, test_loader



