import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.segformer_head import SegformerHead


@MODELS.register_module()
class PanoOptiNetHead(SegformerHead):

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(interpolate_mode=interpolate_mode, **kwargs)

        num_inputs = len(self.in_channels)
        self.convs_feature_template_copy = nn.ModuleList()
        for i in range(num_inputs):
            self.convs_feature_template_copy.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=64,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv_overlap = ConvModule(
            in_channels=64 * num_inputs,
            out_channels=64,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg=None):
        """训练时前向：计算主损失；overlap 仅在存在时加入到 losses。"""
        seg_logits, overlap = self.forward(inputs)

        losses = self.losses(seg_logits, gt_semantic_seg)
        if overlap is not None:
            losses['overlap'] = overlap.float()

        return losses, overlap

    def forward_test(self, inputs, img_metas, test_cfg):
        """测试时前向：必须返回 seg_logits 张量，避免后处理报错。"""
        seg_logits, overlap = self.forward(inputs)
        return seg_logits, overlap

    def predict(self, inputs, batch_img_metas, test_cfg):
        """重写 predict，确保只将 seg_logits 张量传给父类 predict_by_feat()，而非元组。
        
        父类 decode_head.predict() 会调用 predict_by_feat(seg_logits, ...)，
        而 PanoOptiNetHead.forward() 返回 (seg_logits, overlap) 元组，
        直接传进去会导致 resize() 收到 tuple 而报错。
        """
        seg_logits, _ = self.forward(inputs)
        return self.predict_by_feat(seg_logits, batch_img_metas)

    # our PanoOptiNet code: 根据 dx, dy 提供的方向信息找到滑动路径上的重叠的部分
    def featureTemplateCopy(self, x, dx, dy, out_hw_shape):
        if not isinstance(out_hw_shape, (tuple, list)) or len(out_hw_shape) != 2:
            raise ValueError("out_hw_shape 应为包含两个整数的元组或列表 (height, width)")
        if not (dx in [-1, 0, 1] and dy in [-1, 0, 1]) or (dx == 0 and dy == 0):
            raise ValueError("dx 和 dy 必须为 -1、0 或 1，且不能同时为 0")
        if x.ndim != 4:
            raise ValueError("输入特征图 x 应为 4 维张量 (B, C, H, W)")

        B, C, H, W = x.shape
        out_h, out_w = out_hw_shape
        shift_h = out_h // 4
        shift_w = out_w // 4

        region_h = out_h
        region_w = out_w

        direction_slices = {
            (-1, 0): (slice(0, shift_h), slice(0, region_w)),
            (-1, 1): (slice(0, shift_h), slice(region_w - shift_w, region_w)),
            (0, 1): (slice(0, region_h), slice(region_w - shift_w, region_w)),
            (1, 1): (slice(region_h - shift_h, region_h), slice(region_w - shift_w, region_w)),
            (1, 0): (slice(region_h - shift_h, region_h), slice(0, region_w)),
            (1, -1): (slice(region_h - shift_h, region_h), slice(0, shift_w)),
            (0, -1): (slice(0, region_h), slice(0, shift_w)),
            (-1, -1): (slice(0, shift_h), slice(0, shift_w)),
        }

        slice_h, slice_w = direction_slices.get((dx, dy), (None, None))
        if slice_h is None or slice_w is None:
            raise ValueError(f"不支持的滑动方向 dx={dx}, dy={dy}")

        try:
            overlap_data = x[:, :, slice_h, slice_w]
        except Exception as e:
            raise RuntimeError(f"在裁剪重叠区域时发生错误: {e}")

        return overlap_data

    def forward(self, inputs, dx=0, dy=0):
        inputs = self._transform_inputs(inputs)

        outs_overlap = []
        overlap = None

        for idx in range(len(inputs)):
            x_overlap = inputs[idx]
            conv_overlap = self.convs_feature_template_copy[idx]
            outs_overlap.append(
                resize(
                    input=conv_overlap(x_overlap),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out_overlap = self.fusion_conv_overlap(torch.cat(outs_overlap, dim=1))

        if (dx != 0 or dy != 0):
            hw_shape = (out_overlap.shape[2], out_overlap.shape[3])
            overlap = self.featureTemplateCopy(out_overlap, dx, dy, hw_shape)
            overlap = overlap.detach()

        out = self.cls_seg(out)

        return out, overlap

