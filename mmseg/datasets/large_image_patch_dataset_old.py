"""
large_image_patch_dataset.py
============================
LargeImagePatchDataset：在线裁剪数据集，无需预先分割大图

正确用法：
  - 继承 BaseDataset，先设所有自定义属性，再调 super().__init__
  - 实现 load_data_list() 生成 patch 条目（完全不依赖 ann_file）
  - prepare_data() 动态裁剪图像并返回 SegDataSample
"""

import os
from collections import OrderedDict
from typing import List

import numpy as np
from PIL import Image

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class LargeImagePatchDataset(BaseDataset):
    """在线裁剪数据集：直接从 3968×3968 RGBA 大图动态生成 512×512 Patch

    Args:
        images_dir  : 大图目录（相对 data_root 或绝对路径）
        labels_dir  : 标签目录（同名 .png，单通道 0/1/255）
        split_file  : 大图名列表 txt（每行一个，不含扩展名）
        data_root   : 数据根目录
        patch_size  : Patch 尺寸，默认 512
        overlap     : 重叠像素数，默认 128（步长=384）
        img_size    : 大图边长，默认 3968
        max_cache   : LRU 缓存大图数量，默认 8
    """

    METAINFO = dict(
        classes=('background', 'greenland'),
        palette=[[0, 0, 0], [0, 128, 0]]
    )

    def __init__(self,
                 images_dir: str,
                 labels_dir: str,
                 split_file: str,
                 data_root: str = '',
                 patch_size: int = 512,
                 overlap: int = 128,
                 img_size: int = 3968,
                 max_cache: int = 8,
                 metainfo: dict = None,
                 **kwargs):

        # ── 关键：所有自定义属性必须在 super().__init__ 之前设置 ──────────────
        # 因为 super().__init__ → full_init() → load_data_list() 会用到这些属性
        def _p(rel):
            return os.path.join(data_root, rel) if data_root else rel

        self._images_dir = _p(images_dir)
        self._labels_dir = _p(labels_dir)
        self._split_file  = _p(split_file)
        self.patch_size   = patch_size
        self.overlap      = overlap
        self.step         = patch_size - overlap  # 384
        self.img_size     = img_size
        self.max_cache    = max_cache

        assert (img_size - patch_size) % self.step == 0, (
            f"img_size={img_size} - patch_size={patch_size} "
            f"无法被 step={self.step} 整除"
        )
        self.n_patches_1d = (img_size - patch_size) // self.step + 1  # 10

        # LRU 缓存也要提前初始化
        self._img_cache: OrderedDict = OrderedDict()
        self._lbl_cache: OrderedDict = OrderedDict()

        # ── 调用 BaseDataset.__init__（会自动触发 full_init → load_data_list）─
        # serialize_data=False: 我们不序列化 data_list，prepare_data() 自己读图
        super().__init__(
            ann_file='',                    # 不依赖 ann_file，load_data_list 自己处理
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(),
            serialize_data=False,           # 不序列化，patch 数量大
            pipeline=[],                    # 不用 MMSeg pipeline
            lazy_init=False,                # 立即初始化
        )

    # =========================================================================
    #   BaseDataset 核心 override
    # =========================================================================

    def load_data_list(self) -> List[dict]:
        """生成所有 patch 的位置信息列表（不依赖 ann_file）。"""
        with open(self._split_file, 'r') as f:
            large_image_names = [l.strip() for l in f if l.strip()]

        data_list = []
        for name in large_image_names:
            for yi in range(self.n_patches_1d):
                for xi in range(self.n_patches_1d):
                    data_list.append(dict(
                        large_image_name=name,
                        patch_y=yi * self.step,
                        patch_x=xi * self.step,
                        img_path=f'{name}_y{yi*self.step}_x{xi*self.step}.png',
                    ))

        print(f"[LargeImagePatchDataset] "
              f"{len(large_image_names)} 张大图 × {self.n_patches_1d}² "
              f"= {len(data_list)} 个 Patch 样本")
        return data_list

    def prepare_data(self, idx: int) -> dict:
        """动态裁剪图像和标签，返回 MMSeg 标准格式字典。"""
        data_info = self.get_data_info(idx)
        name = data_info['large_image_name']
        y    = data_info['patch_y']
        x    = data_info['patch_x']
        ps   = self.patch_size

        import torch
        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample

        img_large = self._load_image(name)   # [H,W,4] uint8
        lbl_large = self._load_label(name)   # [H,W]   uint8

        img_patch = img_large[y:y+ps, x:x+ps]  # [512,512,4]
        lbl_patch = lbl_large[y:y+ps, x:x+ps]  # [512,512]

        # 分离 RGB 和 NDVI
        rgb  = img_patch[:, :, :3].astype(np.float32)          # [512,512,3]
        ndvi = img_patch[:, :, 3].astype(np.float32) / 255.0   # [512,512] [0,1]

        # [4,512,512]：RGB 保持 float 尺度，NDVI 还原到 [0,255] 供 data_preprocessor 归一化
        rgb_t  = torch.from_numpy(rgb).permute(2, 0, 1)         # [3,512,512]
        ndvi_t = torch.from_numpy(ndvi).unsqueeze(0)            # [1,512,512] [0,1]
        img_t  = torch.cat([rgb_t, ndvi_t * 255.0], dim=0)     # [4,512,512]
        lbl_t  = torch.from_numpy(lbl_patch.astype(np.int64))  # [512,512]

        # 构建 SegDataSample
        ds = SegDataSample()
        ds.set_metainfo(dict(
            img_shape=(ps, ps),
            ori_shape=(ps, ps),
            pad_shape=(ps, ps),
            img_path=data_info['img_path'],
            large_image_name=name,
            patch_y=y,
            patch_x=x,
            ndvi_map=ndvi_t,      # [1,512,512] float [0,1]，供 GMB 使用
        ))
        gt = PixelData()
        gt.data = lbl_t.unsqueeze(0)  # [1,512,512]
        ds.gt_sem_seg = gt

        return dict(inputs=img_t, data_samples=ds)

    # ── LRU 图像缓存 ──────────────────────────────────────────────────────────

    def _load_image(self, name: str) -> np.ndarray:
        if name in self._img_cache:
            self._img_cache.move_to_end(name)
            return self._img_cache[name]
        arr = np.array(Image.open(os.path.join(self._images_dir, name + '.png')))
        if arr.ndim == 2:
            arr = np.stack([arr]*3 + [np.zeros_like(arr)], axis=-1)
        elif arr.shape[2] == 3:
            arr = np.concatenate(
                [arr, np.zeros((*arr.shape[:2], 1), dtype=np.uint8)], axis=2)
        self._img_cache[name] = arr
        if len(self._img_cache) > self.max_cache:
            self._img_cache.popitem(last=False)
        return arr

    def _load_label(self, name: str) -> np.ndarray:
        if name in self._lbl_cache:
            self._lbl_cache.move_to_end(name)
            return self._lbl_cache[name]
        arr = np.array(Image.open(
            os.path.join(self._labels_dir, name + '.png')).convert('L'))
        self._lbl_cache[name] = arr
        if len(self._lbl_cache) > self.max_cache:
            self._lbl_cache.popitem(last=False)
        return arr
