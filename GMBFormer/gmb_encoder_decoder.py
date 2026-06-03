"""
gmb_encoder_decoder.py
=======================
GMBEncoderDecoder：带全局记忆库的 Encoder-Decoder Segmentor

核心职责（相对标准 EncoderDecoder 的改动）：
  1. 从原始 RGBA 输入中分离出 Alpha 通道（NDVI），单独存储
  2. 将 NDVI 图注入 data_samples 的 metainfo，以便 SegformerGMBHead.loss() 能读取
  3. 其余逻辑完全继承自 EncoderDecoder（extract_feat, predict, slide_inference 等）

通道约定：
  - 输入图像必须是 4 通道（RGBA），其中 A 通道 = NDVI（归一化 [0,1]）
  - backbone（MiT-B4）的 in_channels 可以是 4 或 3（如果是 3 则不传 NDVI 给 backbone）
  
实际上：
  - 若 backbone.in_channels == 4：把完整 RGBA 喂给 backbone
  - 若 backbone.in_channels == 3：只把 RGB（前3通道）喂给 backbone，NDVI 只用于 GMB
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                          OptSampleList, SampleList)
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@MODELS.register_module(name='CustomGMBEncoderDecoder')
@MODELS.register_module()
class GMBEncoderDecoder(EncoderDecoder):
    """带全局动态记忆库的 Encoder-Decoder Segmentor

    对 EncoderDecoder 的最小侵入式改造：
      - 重写 extract_feat(): 分离 NDVI，暂存到 self._current_ndvi
      - 重写 loss(): 调用 extract_feat 后将 NDVI 注入 data_samples
      - 重写 encode_decode(): 推理时只用 RGB（或 RGBA）作骨干输入，NDVI 无需传

    Args:
        use_ndvi_channel (bool): 是否从输入中分离 NDVI（Alpha 通道），默认 True
        ndvi_channel_idx (int): NDVI 所在通道索引（0-based），默认 3
        backbone_rgb_only (bool): 若 True，骨干只接收 RGB（前3通道），
                                   NDVI 通道不喂给骨干，只用于 GMB；默认 True
    """

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 use_ndvi_channel: bool = True,
                 ndvi_channel_idx: int = 3,
                 backbone_rgb_only: bool = True):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg
        )
        self.use_ndvi_channel = use_ndvi_channel
        self.ndvi_channel_idx = ndvi_channel_idx
        self.backbone_rgb_only = backbone_rgb_only

        # 临时存储当前 batch 的 NDVI 图，在 loss() 中注入到 data_samples
        self._current_ndvi: Optional[Tensor] = None

    # =========================================================================
    #   分离 NDVI 通道的工具函数
    # =========================================================================
    def _split_ndvi(self, inputs: Tensor):
        """
        从 RGBA 输入中分离出 RGB 和 NDVI。

        Args:
            inputs: [B, 4, H, W]（RGBA，A 通道 = NDVI）

        Returns:
            rgb   : [B, 3, H, W]（前3通道）
            ndvi  : [B, 1, H, W]（Alpha 通道，已归一化 0~1）
        """
        idx = self.ndvi_channel_idx
        # 取除 NDVI 通道外的所有通道作为 RGB（兼容普通3通道）
        channels = list(range(inputs.shape[1]))
        channels.remove(idx)
        rgb  = inputs[:, channels, :, :]
        ndvi = inputs[:, idx:idx+1, :, :]
        return rgb, ndvi

    # =========================================================================
    #   重写 extract_feat：分离 NDVI，根据配置决定喂给骨干的通道
    # =========================================================================
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features，同时分离 NDVI 暂存到 self._current_ndvi。"""
        if self.use_ndvi_channel and inputs.shape[1] > 3:
            rgb, ndvi = self._split_ndvi(inputs)
            self._current_ndvi = ndvi.detach()  # 不参与梯度，只用于 GMB 写入
            backbone_input = rgb if self.backbone_rgb_only else inputs
        else:
            backbone_input = inputs
            self._current_ndvi = None

        x = self.backbone(backbone_input)
        if self.with_neck:
            x = self.neck(x)
        return x

    # =========================================================================
    #   重写 loss：在训练时将 NDVI 注入 data_samples
    # =========================================================================
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """
        训练前向：
          1. extract_feat → 分离 NDVI 到 self._current_ndvi
          2. 将 NDVI 图注入 data_samples[i].metainfo['ndvi_map']
          3. 调用 decode_head.loss()（SegformerGMBHead 会接收 NDVI）
        """
        x = self.extract_feat(inputs)

        # 将 NDVI 注入 data_samples（SegformerGMBHead.loss() 里会读取）
        if self._current_ndvi is not None:
            for i, ds in enumerate(data_samples):
                if not hasattr(ds, 'metainfo'):
                    ds.metainfo = {}
                ds.metainfo['ndvi_map'] = self._current_ndvi[i]  # [1, H, W]

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
