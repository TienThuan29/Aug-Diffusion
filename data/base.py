from PIL import Image
from torchvision import transforms
from .transform import RandomHFlip, RandomVFlip, RandomRotation

class TrainBaseTransform:
    """
        Resize, flip, rotation for image and mask
    """
    def __init__(
            self,
            input_size,
            hflip: bool,
            vflip: bool,
            rotate: bool,
    ):
        self.input_size = input_size
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate

    def __call__(self, image, mask):
        transforms_fn = transforms.Resize(self.input_size, Image.BILINEAR)
        image = transforms_fn(image)
        transforms_fn = transforms.Resize(self.input_size, Image.NEAREST)
        mask = transforms_fn(mask)

        if self.hflip:
            transforms_fn = RandomHFlip()
            image, mask = transforms_fn(image, mask)
        if self.vflip:
            transforms_fn = RandomVFlip()
            image, mask = transforms_fn(image, mask)
        if self.rotate:
            transform_fn = RandomRotation([0, 90, 180, 270])
            image, mask = transform_fn(image, mask)

        return image, mask


class TestBaseTransform:
    """ Resize image, mask """
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, image, mask):
        transform_fn = transforms.Resize(self.input_size, Image.BILINEAR)
        image = transform_fn(image)
        transform_fn = transforms.Resize(self.input_size, Image.NEAREST)
        mask = transform_fn(mask)
        return image, mask
  