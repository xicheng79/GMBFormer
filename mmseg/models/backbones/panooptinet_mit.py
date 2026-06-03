# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmseg.registry import MODELS
from ..utils import nlc_to_nchw
from mmseg.models.backbones.mit import MixVisionTransformer


@MODELS.register_module()
class PanoOptiNetMixVisionTransformer(MixVisionTransformer):
    """PanoOptiNet 的主干网络。

    该主干网络实现了 `PanoOptiNet`，基于 SegFormer 的设计，并增加了滑动窗口的重叠处理。
    该类继承自 MixVisionTransformer，并添加了处理重叠区域的功能。

    参数:
        in_channels (int): 输入通道数。默认值：3。
        embed_dims (int): 嵌入维度。默认值：64。
        num_stages (int): 阶段数量。默认值：4。
        num_layers (Sequence[int]): 每个 Transformer 编码层的层数。
            默认值：[3, 4, 6, 3]。
        num_heads (Sequence[int]): 每个 Transformer 编码层的注意力头数。
            默认值：[1, 2, 4, 8]。
        patch_sizes (Sequence[int]): 每个重叠补丁嵌入的补丁大小。
            默认值：[7, 3, 3, 3]。
        strides (Sequence[int]): 每个重叠补丁嵌入的步长。
            默认值：[4, 2, 2, 2]。
        sr_ratios (Sequence[int]): 每个 Transformer 编码层的空间降维比率。
            默认值：[8, 4, 2, 1]。
        out_indices (Sequence[int] | int): 指定输出哪些阶段的特征。
            默认值：(0, 1, 2, 3)。
        mlp_ratio (int): MLP 隐藏层维度与嵌入维度的比例。
            默认值：4。
        qkv_bias (bool): 是否为 QKV 启用偏置。默认值：True。
        drop_rate (float): 元素被置零的概率。默认值：0.0。
        attn_drop_rate (float): 注意力层的 dropout 率。默认值：0.0。
        drop_path_rate (float): 随机深度率。默认值：0.0。
        norm_cfg (dict): 归一化层的配置字典。
            默认值：dict(type='LN')。
        act_cfg (dict): FFN 的激活函数配置。
            默认值：dict(type='GELU')。
        pretrained (str, optional): 预训练模型路径。默认值：None。
        init_cfg (dict or list[dict], optional): 初始化配置字典。
            默认值：None。
        with_cp (bool): 是否使用检查点。使用检查点可以节省部分显存，
            但会降低训练速度。默认值：False。
    """

    def __init__(
            self,
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 4, 6, 3],
            num_heads=[1, 2, 4, 8],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            act_cfg=dict(type="GELU"),
            norm_cfg=dict(type="LN", eps=1e-6),
            pretrained=None,
            init_cfg=None,
            with_cp=False,
    ):
        # 直接调用父类构造，所有结构超参都由 config 动态传入
        super(PanoOptiNetMixVisionTransformer, self).__init__(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_stages=num_stages,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_sizes=patch_sizes,
            strides=strides,
            sr_ratios=sr_ratios,
            out_indices=out_indices,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            with_cp=with_cp
        )

        # PanoOptiNet 专用缓存
        self.last_dx = None
        self.last_dy = None

    # our PanoOptiNet code: 根据 dx, dy 提供的方向信息将重叠的部分复制到对应的位置
    def featureTemplatePaste(self, x, overlap_data, dx, dy, out_hw_shape):
        """
        根据方向信息 `(dx, dy)`，将重叠部分的数据复制到特征图的对应位置。
        参数:
            x (Tensor): 输入特征图，形状为 (batch, channels, H, W)。
            overlap_data (Tensor): 重叠部分的数据，形状与目标区域一致。
            dx (int): x 方向上的偏移量。
            dy (int): y 方向上的偏移量。
            out_hw_shape (tuple): 输出特征图的空间形状 (H, W)。
        返回:
            Tensor: 处理后的特征图，形状为 (batch, channels, H, W)。
        异常:
            ValueError: 如果 `(dx, dy)` 不合法，或者 `overlap_data` 的形状与目标区域不匹配。
        """
        # 下面的大小全部依据当前层的 out_hw_shape 动态计算，不写死常数
        shift_size = int(out_hw_shape[0] / 4)
        region_size = int(out_hw_shape[0])

        dx_dy_to_slice = {
            (-1, 0): (slice(0, shift_size), slice(0, region_size)),
            (-1, 1): (slice(0, shift_size),
                      slice(region_size - shift_size, region_size)),
            (0, 1): (slice(0, region_size),
                     slice(region_size - shift_size, region_size)),
            (1, 1): (slice(region_size - shift_size, region_size),
                     slice(region_size - shift_size, region_size)),
            (1, 0): (slice(region_size - shift_size, region_size),
                     slice(0, region_size)),
            (1, -1): (slice(region_size - shift_size, region_size),
                      slice(0, shift_size)),
            (0, -1): (slice(0, region_size), slice(0, shift_size)),
            (-1, -1): (slice(0, shift_size), slice(0, shift_size)),
        }
        if (dx, dy) not in dx_dy_to_slice:
            raise ValueError(f"Unsupported dx={dx} and dy={dy} combination.")

        target_slice = dx_dy_to_slice[(dx, dy)]
        target_region = x[:, :, target_slice[0], target_slice[1]]
        if overlap_data.shape != target_region.shape:
            raise ValueError(
                "Overlap data shape does not match target region: "
                f"{overlap_data.shape} vs {target_region.shape}"
            )

        x[:, :, target_slice[0], target_slice[1]] = overlap_data
        return x

    def forward(self, x, img_metas=None, overlap=None):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入特征，形状为 (batch, channels, H, W)。
            img_metas (list): 图像的元信息，包含文件名等信息。
            overlap (Tensor): 重叠部分的数据，用于 PanoOptiNet 的自定义处理。
        返回:
            list: 输出特征列表，每个元素的形状为 (batch, channels, H, W)。
        """
        # 典型 MMSeg 训练/推理路径：只传 x
        if img_metas is None:
            # 完全走父类的动态实现，不做任何固定尺寸假设
            return super(PanoOptiNetMixVisionTransformer, self).forward(x)

        # 自定义路径：需要从文件名中解析 dx, dy, id
        filename = img_metas[0]["filename"]
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        dx = int(filename.split("_")[-2])
        dy = int(filename.split("_")[-1])
        tile_id = int(filename.split("_")[-3])

        outs = []
        # self.layers 是父类构造的每个 stage 的 (patch_embed, block, norm) 组合
        for i, layer in enumerate(self.layers):
            # patch embedding（动态输出 hw_shape）
            x, hw_shape = layer[0](x)  # hw_shape 由当前输入实时计算得到；不做任何写死

            # 多个 transformer blocks（注意：所有注意力/窗口大小均由父类依据 hw_shape 动态处理）
            for block in layer[1]:
                x = block(x, hw_shape)

            # 层归一化
            x = layer[2](x)

            # (N, H*W, C) -> (N, C, H, W)
            x = nlc_to_nchw(x, hw_shape)

            # PanoOptiNet：仅在第 0 层、且满足条件时粘贴重叠区域
            if (dx != 0 or dy != 0) and tile_id != 1 and i == 0:
                if overlap is not None and overlap != []:
                    x = self.featureTemplatePaste(
                        x, overlap, self.last_dx, self.last_dy, hw_shape
                    )

            if i in self.out_indices:
                outs.append(x)

        # 记录当前偏移，供下一次使用
        self.last_dx = dx
        self.last_dy = dy

        return outs