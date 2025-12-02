import json 
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from .base import TrainBaseTransform, TestBaseTransform
from .transform import RandomColorJitter
from .image_reader import build_image_reader


def build_custom_dataset(cfg, training, class_name=None):
      image_reader = build_image_reader(cfg.image_reader)
      normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
      
      if training:
            transform_fn = TrainBaseTransform(
                  cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
            )
      else:
            transform_fn = TestBaseTransform(cfg["input_size"])
      
      colorjitter_fn = None
      if cfg.get("colorjitter", None) and training:
            colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

      print("building CustomDataset from: {}".format(cfg["meta_file"]))
      
      dataset = CustomDataset(
            image_reader,
            cfg["meta_file"],
            training,
            transform_fn=transform_fn,
            normalize_fn=normalize_fn,
            colorjitter_fn=colorjitter_fn,
            class_name=class_name,
      )
      
      sampler = RandomSampler(dataset)
      
      data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
      )
      return data_loader


class CustomDataset(Dataset):
      def __init__(
            self,
            image_reader,
            meta_file,
            training,
            transform_fn,
            normalize_fn,
            colorjitter_fn=None,
            class_name=None
      ):
            super().__init__()
            self.image_reader = image_reader
            self.meta_file = meta_file
            self.training = training
            self.transform_fn = transform_fn
            # self.normalize_fn = normalize_fn
            self.colorjitter_fn = colorjitter_fn
            self.class_name = class_name
            
            # construct metas
            with open(meta_file, "r") as f_r:
                  self.metas = []
                  for line in f_r:
                        meta =  json.loads(line)
                        # filter by class_name if specified
                        if class_name is not None:
                              # get class from filename or clsname field
                              item_class =  meta.get("clsname", meta["filename"].split("/")[-4])
                              if item_class != class_name:
                                    continue
                        self.metas.append(meta)
            
            print(f"Dataset loaded: {len(self.metas)} samples" + 
                   (f" (filtered by class: {class_name})" if class_name else ""))
      
      
      def __len__(self):
            return len(self.metas)
      
      def __getitem__(self, index):
            input = {}
            meta = self.metas[index]
            # read image
            filename = meta["filename"]
            label = meta["label"]
            image = self.image_reader(meta["filename"])
            input.update({
                  "filename": filename,
                  "height": image.shape[0],
                  "width": image.shape[1],
                  "label": label
            })
            
            if meta.get("clsname", None):
                  input["clsname"] = meta["clsname"]
            else:
                  input["clsname"] = filename.split("/")[-4]
            
            image = Image.fromarray(image, "RGB")
            
            # read / generate mask
            if meta.get("mask", None):
                  mask = self.image_reader(meta["maskname"], is_mask=True)
            else:
                  if label == 0:  # good
                        mask = np.zeros((image.height, image.width)).astype(np.uint8)
                  elif label == 1:  # defective
                        mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
                  else:
                        raise ValueError("Labels must be [None, 0, 1]!")
            
            mask = Image.fromarray(mask, "L")
            
            if self.transform_fn:
                  image, mask = self.transform_fn(image, mask)
            
            if self.colorjitter_fn:
                  image = self.colorjitter_fn(image)
            
            # convert np.ndarray to tensor for torch model
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
            
            # if self.normalize_fn:
            #       image = self.normalize_fn(image)
            
            input.update({
                  "image": image,
                  "mask": mask
            })
            return input