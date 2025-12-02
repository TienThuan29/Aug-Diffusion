import os
import cv2

class ImageReader:
    def __init__(self, image_dir: str, color_mode: str):
        self.image_dir = image_dir
        self.color_mode = color_mode
        self.color_modes = ['RGB', 'BGR', 'GRAY']
        assert color_mode in self.color_modes, f'{color_mode} is not support'
        if color_mode != "BGR":
            self.cvt_color = getattr(cv2, f"COLOR_BGR2{color_mode}")
        else:
            self.cvt_color = None

    def __call__(self, filename: str, is_mask: bool=False):
        filename = os.path.join(self.image_dir, filename)
        assert os.path.exists(filename), f'{filename} does not exist'
        if is_mask:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return image # mask image
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.color_mode != "BGR":
            img = cv2.cvtColor(image, self.cvt_color)
        return image # color image

def build_image_reader(cfg_reader):
    if cfg_reader["type"] == "opencv":
        return ImageReader(**cfg_reader["kwargs"])
    else:
        raise TypeError("no supported image reader type: {}".format(cfg_reader["type"]))
