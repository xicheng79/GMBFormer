"""
segformer_gmb_head.py
=====================
SegformerGMBHead: 带全局动态记忆库 (Global Memory Bank) 的 SegFormer 解码头

核心改动（相对原版 SegformerHead）：
  1. 在 Stage-3（C3）特征进入 MLP 前，插入 GlobalMemoryBankModule 做增强
  2. 重写 loss() 方法，在训练时将 NDVI 图传给 GlobalMemoryBankModule 用于记忆写入
  3. 其余逻辑（多级 MLP + Upsample + Concat + FusionConv + ClsSeg）与原版完全相同

NDVI 传递约定：
  - data_samples 的 metainfo 中存放 ndvi_map（由 Dataset 的 pipeline 放入）
  - 或者由 Segmentor 从原始 RGBA 输入中分离出 Alpha 通道后传入
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from .global_memory_bank import GlobalMemoryBankModule
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import SampleList


@MODELS.register_module(name='CustomSegformerGMBHead')
@MODELS.register_module()
class SegformerGMBHead(BaseDecodeHead):
    """带全局动态记忆库的 SegFormer 解码头

    对 SegformerHead 的最小侵入式改造：
      - 只在 C3 特征路径上插入 GlobalMemoryBankModule
      - 其余特征处理路径（C1,C2,C4）保持原样

    Args:
        memory_size (int): 记忆库槽位数（绿地原型数量），默认 64
        momentum (float): 记忆库 EMA 更新动量，默认 0.99
        ndvi_thresh (float): NDVI 均值阈值，超过才写入记忆库，默认 0.2
        memory_heads (int): Cross-Attention 多头数，默认 8
        ndvi_channel_idx (int): NDVI 所在的输入通道索引（0-based），默认 3（Alpha）
        interpolate_mode (str): MLP 上采样模式，默认 'bilinear'
        **kwargs: 其余参数透传给 BaseDecodeHead
    """

    def __init__(self,
                 memory_size: int = 64,
                 momentum: float = 0.99,
                 ndvi_thresh: float = 0.6,
                 memory_heads: int = 8,
                 ndvi_channel_idx: int = 3,
                 interpolate_mode: str = 'bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.ndvi_channel_idx = ndvi_channel_idx

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        # ── 原版 SegformerHead 的 MLP conv 层 ─────────────────────────────
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        # ── 全局记忆库模块（插在 C3 特征路径上）────────────────────────────
        # in_channels[-2] 是 C3 的通道数（MiT-B4 中为 320）
        self.gmb = GlobalMemoryBankModule(
            in_channels=self.in_channels[-2],
            memory_size=memory_size,
            momentum=momentum,
            ndvi_thresh=ndvi_thresh,
            num_heads=memory_heads
        )

    # =========================================================================
    #   forward：训练 / 推理共用，ndvi_maps 仅训练时有效
    # =========================================================================
    def forward(self,
                inputs: Tuple[torch.Tensor],
                ndvi_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs    : (C1, C2, C3, C4) 四级特征图 tuple
            ndvi_maps : [B, 1, H, W]，从原始输入 Alpha 通道提取的 NDVI 图
                        推理时不需要传，传 None 即可

        Returns:
            seg_logits: [B, num_classes, H/4, W/4]
        """
        inputs = self._transform_inputs(inputs)  # 选出 in_index 对应的特征
        # inputs 是一个 list: [C1, C2, C3, C4]

        # ── C3 记忆库增强（语义已基本成形，分辨率比 C4 大 4 倍）─────────────
        # 索引 -2 对应 in_index 的倒数第二项，即 C3
        c3_enhanced = self.gmb(inputs[-2], ndvi_maps)

        # 替换 C3，其余不变
        inputs_enhanced = list(inputs)
        inputs_enhanced[-2] = c3_enhanced

        # ── 原版 SegformerHead 逻辑（多级 MLP + Upsample + Concat + Fuse）─
        outs = []
        for idx in range(len(inputs_enhanced)):
            x = inputs_enhanced[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs_enhanced[0].shape[2:],  # 上采样到 C1 的分辨率 (1/4)
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)
        return out

    # =========================================================================
    #   loss：重写以在训练时传入 NDVI 图
    # =========================================================================
    def loss(self,
             inputs: Tuple[torch.Tensor],
             batch_data_samples: SampleList,
             train_cfg) -> dict:
        """
        重写 BaseDecodeHead.loss()，主要目的是从 data_samples 里取出 NDVI 图
        并传给 forward()，供 GlobalMemoryBankModule 做记忆写入。

        NDVI 来源：data_samples[i].metainfo.get('ndvi_map')
        也可由 GMBEncoderDecoder 在调用前注入到 data_samples 里。
        """
        # 收集 NDVI 图（如果 data_samples 携带了的话）
        ndvi_maps = self._collect_ndvi(batch_data_samples, inputs[0].device)

        # 前向传播（含 GMB 增强和记忆库写入）
        seg_logits = self.forward(inputs, ndvi_maps)

        # 计算损失（调用基类）
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def _collect_ndvi(self,
                      batch_data_samples: SampleList,
                      device: torch.device) -> Optional[torch.Tensor]:
        """
        从 data_samples 的 metainfo 中提取 NDVI 图并堆叠为 batch tensor。

        如果 data_samples 中没有 ndvi_map，返回 None（不影响推理）。
        """
        ndvi_list = []
        for ds in batch_data_samples:
            ndvi = ds.metainfo.get('ndvi_map', None)
            if ndvi is None:
                return None  # 只要有一个没有，就不用 NDVI
            if not isinstance(ndvi, torch.Tensor):
                ndvi = torch.tensor(ndvi, dtype=torch.float32)
            ndvi_list.append(ndvi)

        if not ndvi_list:
            return None

        # 堆叠为 [B, 1, H, W]
        ndvi_batch = torch.stack(ndvi_list, dim=0).to(device)
        if ndvi_batch.dim() == 3:
            ndvi_batch = ndvi_batch.unsqueeze(1)
        return ndvi_batch
