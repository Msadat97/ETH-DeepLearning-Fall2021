import io
import pathlib
from typing import TypeVar, Union, Optional, Union
from enum import Enum, auto
import numpy as np
import torch
import cv2
import einops
import kornia

from .base import BaseZipSaver

#--------------------------------------------------------------------------
# Type aliases
#--------------------------------------------------------------------------

# path types: str or pathlib.Path
_Path = TypeVar("_Path", str, pathlib.Path)

# Image types: numpy ndarray to torch Tensor
_Image = TypeVar("_Image", torch.Tensor, np.ndarray)

#--------------------------------------------------------------------------
# Image io utilities
#--------------------------------------------------------------------------

def load_image(fpath: _Path, mode: str = "RGB") -> np.ndarray:
    if not mode in ["RGB", "BGR"]:
        raise ValueError(f"{mode} is not supported")
    image = cv2.imread(str(fpath))
    if mode == "RGB":
        image = from_cv2(image)
    return image


def save_image(fpath: _Path, image: np.ndarray, input_mode: str = "RGB") -> None:
    if not input_mode.upper() in ["RGB", "BGR", "Gray"]:
        raise ValueError(f"{input_mode} is not supported")
    if input_mode == "RGB":
        image = to_cv2(image)
    cv2.imwrite(str(fpath), image)


def save_tensor_image(fpath: _Path, image: torch.Tensor, drange: tuple = (0, 255)) -> None:
    image = from_torch(image, drange)
    save_image(fpath, image)


def image_to_bytes(image: np.ndarray, mode: str = "RGB") -> bytes:
    if mode.upper() == "RGB":
        image = to_cv2(image)
    success, buffer = cv2.imencode(".png", image)
    if success:
        image_bytes = io.BytesIO(buffer)
    else:
        raise Exception("not abale to convert the image to binary")
    return image_bytes.getvalue()


class ImageZipSaver(BaseZipSaver):
    "interface for saving a list of images to a zipfile incrementally"
    def __init__(
        self, fpath: Union[str, pathlib.Path], basename: Optional[str] = "data") -> None:
        super().__init__(fpath, basename=basename)
        self.save_lib = save_lib

    def save_file(self, image: np.ndarray) -> None:
        image_bytes = image_to_bytes(image)
        self.zip_file.writestr(self.get_current_name() + ".png", image_bytes)

#--------------------------------------------------------------------------
# Conversion utilities
#--------------------------------------------------------------------------

def from_torch(image: torch.Tensor, drange: tuple = (0, 255)) -> np.ndarray:
    img = kornia.tensor_to_image(image)
    img = change_drange(img, drange[0], drange[1])
    return to_uint8(img)

def to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.rint(image).clip(0, 255).astype("uint8")
    return image

def to_cv2(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def from_cv2(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def change_drange(image: _Image, dlow: int = 0, dhigh: int = 255) -> _Image:
    return (image - dlow) / (dhigh - dlow) * 255

#--------------------------------------------------------------------------
        
def gen_lut():
    """
    Generate a label colormap compatible with opencv lookup table, based on
    Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
    appendix C2 `Pseudocolor Generation`.
    :Returns:
        color_lut : opencv compatible color lookup table
    """
    tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
    arr = np.arange(256)
    r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
    g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
    b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
    return np.concatenate([[[b]], [[g]], [[r]]]).T

def labels2rgb(labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Convert a label image to an rgb image using a lookup table
    :Parameters:
        labels : an image of type np.uint8 2D array
        lut : a lookup table of shape (256, 3) and type np.uint8
    :Returns:
        colorized_labels : a colorized label image
    """
    return cv2.LUT(cv2.merge((labels, labels, labels)), lut)
