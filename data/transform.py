import torch
import numbers
import random
from typing import Tuple
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
)

class RandomHFlip:
    def __init__(self, flip_p=0.5):
        self.flip_p = flip_p

    def __call__(self, image, mask):
        flip_flag = torch.rand(1)[0].item()< self.flip_p
        if flip_flag:
            return F.hflip(image), F.hflip(mask)
        else:
            return image, mask


class RandomVFlip:
    def __init__(self, flip_p=0.5):
        self.flip_p = flip_p

    def __call__(self, image, mask):
        flip_flag = torch.rand(1)[0].item()< self.flip_p
        if flip_flag:
            return F.vflip(image), F.vflip(mask)
        else:
            return image, mask


class RandomRotation:
    def __init__(
            self,
            degrees,
            resample: bool = False,
            expand: bool = False,
            center = None,
    ):
        if isinstance(degrees, numbers.Number):
            degrees = [degrees]
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """
        Lấy tham số cho ``rotate`` để quay ngẫu nhiên.
        return: 1 số ngẫu nhiên trong degrees
        """
        angle = random.choice(degrees)
        return angle

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        image, mask (PIL image)
        return: rotated image, rotated mask
        """
        angle = self.get_params(self.degrees)
        image = F.rotate(image, angle, self.resample, self.expand, self.center)
        mask = F.rotate(mask, angle, self.resample, self.expand, self.center)
        return image, mask

    def __repr__(self):
        """
        hàm này trả về chuỗi biểu diễn của đối,
        sử dụng trong logging/debug
        """
        format_string = self.__class__.__name__ + "(degrees={0}".format(self.degrees)
        format_string += ", resample={0}".format(self.resample)
        format_string += ", expand={0}".format(self.expand)
        if self.center is not None:
            format_string += ", center={0}".format(self.center)
        format_string += ")"
        return format_string


class RandomColorJitter:

    def __init__(self,brightness: int = 0, contrast: int = 0, saturation: int = 0, hue: int = 0,prop: int = 0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.prop = prop
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )


    def _check_input(self, value, name, center, bound, clip_first_on_zero):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )
        if value[0] == value[1] == center:
            value = None
        return value


    def get_params(self, brightness, contrast, saturation, hue):
        """
        Lấy một phép biến đổi ngẫu nhiên để áp dụng cho hình ảnh.

        Các đối số giống với __init__.

        return:
        Phép biến đổi điều chỉnh ngẫu nhiên độ sáng, độ tương phản và
        độ bão hòa theo thứ tự ngẫu nhiên.
        """
        img_transforms = []

        if brightness is not None and random.random() < self.prob:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_brightness(img, brightness_factor))
            )

        if contrast is not None and random.random() < self.prob:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_contrast(img, contrast_factor))
            )

        if saturation is not None and random.random() < self.prob:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_saturation(img, saturation_factor))
            )

        if hue is not None and random.random() < self.prob:
            hue_factor = random.uniform(hue[0], hue[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_hue(img, hue_factor))
            )

        random.shuffle(img_transforms)
        img_transforms = transforms.Compose(img_transforms)

        return img_transforms
