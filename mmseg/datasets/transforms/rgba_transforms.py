# Copyright (c) OpenMMLab. All rights reserved.
import io
import warnings
from typing import Optional, Sequence

import cv2
import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from numpy import random

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None


@DATASETS.register_module()
class SmallPatchDataset(BaseSegDataset):
    """VOC-style small patch dataset for RGB+NDVI segmentation."""

    METAINFO = dict(
        classes=('background', 'low_vegetation', 'tree'),
        palette=[[0, 0, 0], [0, 255, 255], [0, 255, 0]])

    def __init__(self,
                 ann_file: str = '',
                 img_suffix: str = '.png',
                 seg_map_suffix: str = '.png',
                 **kwargs) -> None:
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)


@TRANSFORMS.register_module()
class LoadRGBAImageFromFile(BaseTransform):
    """Load an image as four channels: RGB plus alpha/NDVI."""

    def __init__(self,
                 to_float32: bool = False,
                 imdecode_backend: str = 'pillow',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead',
                DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')
            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _decode_rgba(self, img_bytes: bytes) -> np.ndarray:
        if self.imdecode_backend == 'pillow' and Image is not None:
            with io.BytesIO(img_bytes) as buff:
                img = Image.open(buff)
                img = ImageOps.exif_transpose(img)
                return np.array(img.convert('RGBA'))

        img = mmcv.imfrombytes(
            img_bytes, flag='unchanged', backend=self.imdecode_backend)
        if img.ndim == 2:
            rgb = np.repeat(img[..., None], 3, axis=2)
            alpha = np.full_like(img[..., None], 255)
            return np.concatenate([rgb, alpha], axis=2)

        channels = img.shape[2]
        if channels == 1:
            rgb = np.repeat(img, 3, axis=2)
            alpha = np.full_like(img, 255)
            return np.concatenate([rgb, alpha], axis=2)
        if channels == 3:
            rgb = img[..., ::-1]
            alpha = np.full(img.shape[:2] + (1, ), 255, dtype=img.dtype)
            return np.concatenate([rgb, alpha], axis=2)
        if channels >= 4:
            return img[..., [2, 1, 0, 3]]
        raise ValueError(f'Unsupported image shape: {img.shape}')

    def transform(self, results: dict) -> Optional[dict]:
        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = self._decode_rgba(img_bytes)
        except Exception as e:
            if self.ignore_empty:
                return None
            raise e

        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'ignore_empty={self.ignore_empty}, '
                f'to_float32={self.to_float32}, '
                f"imdecode_backend='{self.imdecode_backend}', "
                f'backend_args={self.backend_args})')


@TRANSFORMS.register_module()
class PhotoMetricDistortionRGBA(BaseTransform):
    """Apply photometric distortion to RGB channels and preserve alpha."""

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self,
                img: np.ndarray,
                alpha: float = 1,
                beta: float = 0) -> np.ndarray:
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower,
                                     self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        assert img.ndim == 3 and img.shape[2] in (3, 4), (
            'PhotoMetricDistortionRGBA expects an RGB/RGBA image, '
            f'but got shape {img.shape}')

        rgb = img[..., :3]
        alpha = img[..., 3:] if img.shape[2] == 4 else None

        rgb = self.brightness(rgb)
        mode = random.randint(2)
        if mode == 1:
            rgb = self.contrast(rgb)
        rgb = self.saturation(rgb)
        rgb = self.hue(rgb)
        if mode == 0:
            rgb = self.contrast(rgb)

        if alpha is not None:
            img = np.concatenate([rgb, alpha.astype(rgb.dtype)], axis=2)
        else:
            img = rgb
        results['img'] = img
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'brightness_delta={self.brightness_delta}, '
                f'contrast_range=({self.contrast_lower}, '
                f'{self.contrast_upper}), '
                f'saturation_range=({self.saturation_lower}, '
                f'{self.saturation_upper}), '
                f'hue_delta={self.hue_delta})')


__all__ = [
    'SmallPatchDataset', 'LoadRGBAImageFromFile', 'PhotoMetricDistortionRGBA'
]
