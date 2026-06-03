# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
# 只导入 MixVisionTransformer
from .mit import MixVisionTransformer

class SimpleCNNBlock(BaseModule):
    """轻量级光谱辅流的一个 Stage"""
    def __init__(self, in_channels, out_channels, stride=1, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        # 使用步长来实现下采样，保持与主路一致的空间尺寸
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class FusionBlock(BaseModule):
    """简单的融合模块：将光谱特征投影后加到 RGB 特征上"""
    def __init__(self, rgb_dim, spec_dim, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        # 1x1 卷积用于对齐通道数
        self.proj = ConvModule(
            spec_dim,
            rgb_dim,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None) # 融合时不加激活函数

    def forward(self, x_rgb, x_spec):
        # x_rgb: (B, C_rgb, H, W) (4D Tensor)
        # x_spec: (B, C_spec, H, W) (4D Tensor)
        x_spec_projected = self.proj(x_spec)
        # 简单的相加融合
        return x_rgb + x_spec_projected

@MODELS.register_module()
class DualStreamMiT(MixVisionTransformer):
    """
    DSSF-SegFormer 的双流骨干网络 (修复版 v6：正确传递 hw_shape)。
    """
    def __init__(self, 
                 spec_in_channels=1, 
                 spec_embed_dims=[32, 64, 128, 256],
                 **kwargs):
        # ----------------------------------------------------------
        # 1. 初始化 RGB 主流 (调用父类)
        # ----------------------------------------------------------
        # 强制告诉父类输入是 3 通道 RGB
        kwargs['in_channels'] = 3
        super().__init__(**kwargs)
        
        # 此时父类已经创建了 self.layers，它是一个 ModuleList，包含了4个阶段
        # 每个 stage 也是一个 ModuleList，结构通常是 [patch_embed, blocks, norm]

        # ----------------------------------------------------------
        # 2. 初始化光谱辅流 和 融合模块
        # ----------------------------------------------------------
        self.spec_in_channels = spec_in_channels
        self.norm_cfg = dict(type='SyncBN', requires_grad=True)
        
        # 确保 num_stages 属性存在
        if not hasattr(self, 'num_stages'):
             self.num_stages = len(self.layers)

        # A. 构建光谱辅流 (Spectral Auxiliary Stream) - 轻量级 CNN
        self.spec_stages = nn.ModuleList()
        # Stage 1: 下采样 4 倍
        self.spec_stages.append(SimpleCNNBlock(spec_in_channels, spec_embed_dims[0], stride=4, norm_cfg=self.norm_cfg))
        # Stage 2, 3, 4: 下采样 2 倍
        for i in range(1, self.num_stages):
            self.spec_stages.append(SimpleCNNBlock(spec_embed_dims[i-1], spec_embed_dims[i], stride=2, norm_cfg=self.norm_cfg))

        # B. 构建融合模块 (Fusion Blocks)
        self.fusion_blocks = nn.ModuleList()
        
        # 【可靠地获取 RGB 主流每个阶段的维度】
        rgb_stage_dims = []
        for i in range(self.num_stages):
             # 直接访问 PatchEmbed 模块的 embed_dims 属性获取输出维度
             rgb_stage_dims.append(self.layers[i][0].embed_dims)

        for i in range(self.num_stages):
            self.fusion_blocks.append(
                FusionBlock(rgb_dim=rgb_stage_dims[i], spec_dim=spec_embed_dims[i], norm_cfg=self.norm_cfg)
            )

    # def forward(self, x):
    #     """
    #     完整的双流前向传播逻辑
    #     x: (B, 4, H, W) - RGBA 四通道输入
    #     """
    #     # 1. 拆分输入
    #     x_rgb = x[:, :3, :, :]
    #     x_spec = x[:, 3:, :, :]

    #     outs = []
    #     # rgb_feat 初始为图像输入 (B, 3, H, W)
    #     rgb_feat = x_rgb
    #     spec_feat = x_spec

    #     # 循环处理 4 个阶段
    #     for i in range(self.num_stages):
    #         # ===========================
    #         # A. 光谱流 (Auxiliary Stream)
    #         # ===========================
    #         # 通过 CNN Block，输出 4D 张量
    #         spec_feat = self.spec_stages[i](spec_feat)

    #         # ===========================
    #         # B. RGB流 (Main Stream) - 获取父类组件
    #         # ===========================
    #         # 从 self.layers 中解包当前阶段的组件
    #         # 标准 MiT 结构：[patch_embed, blocks, norm]
    #         patch_embed, blocks, norm = self.layers[i]

    #         # ===========================
    #         # C. RGB流 - Patch Embedding
    #         # ===========================
    #         # MMseg 1.x 中 patch_embed 返回 (tokens, (H, W))
    #         # 使用嵌套解包正确获取 H 和 W
    #         rgb_feat_tokens, (H, W) = patch_embed(rgb_feat)

    #         # ===========================
    #         # D. 融合 (Fusion)
    #         # ===========================
    #         # 融合模块需要 4D 张量 (B, C, H, W) 作为输入
    #         # 将 Transformer 的 tokens 转回 4D 张量
    #         rgb_feat_4d = rgb_feat_tokens.reshape(
    #             rgb_feat_tokens.shape[0], H, W, rgb_feat_tokens.shape[-1]).permute(0, 3, 1, 2)
            
    #         # 执行融合：RGB 4D + 投影后的光谱 4D
    #         fused_feat_4d = self.fusion_blocks[i](rgb_feat_4d, spec_feat)
            
    #         # 将融合后的特征转回 tokens (B, N, C) 给 Transformer Block 使用
    #         rgb_feat_tokens = fused_feat_4d.flatten(2).transpose(1, 2)

    #         # ===========================
    #         # E. RGB流 - Transformer Blocks & Norm
    #         # ===========================
    #         # 输入输出都是 tokens (B, N, C)
    #         for blk in blocks:
    #             # 【关键修复】MMSeg 1.x 的 TransformerBlock forward 需要 hw_shape 参数
    #             # 将 H 和 W 打包成元组传入
    #             rgb_feat_tokens = blk(rgb_feat_tokens, hw_shape=(H, W))
                
    #         rgb_feat_tokens = norm(rgb_feat_tokens)

    #         # ===========================
    #         # F. 输出准备
    #         # ===========================
    #         # 将最终 tokens 转回 4D 张量，作为本阶段输出，以及下一阶段的输入
    #         rgb_feat_out = rgb_feat_tokens.reshape(
    #             rgb_feat_tokens.shape[0], H, W, rgb_feat_tokens.shape[-1]).permute(0, 3, 1, 2)
            
    #         outs.append(rgb_feat_out)
    #         # 重要：更新 rgb_feat 为当前阶段输出的 4D 张量，用于下一阶段的 Patch Embed 输入
    #         rgb_feat = rgb_feat_out 

    #     return outs
    
    #将深层光谱特征（ spec_feat_s4 ，语义信息丰富）与浅层RGB特征（ rgb_feat_s3 ，空间信息丰富）进行融合
    def forward(self, x):
        """
        完整的双流前向传播逻辑
        x: (B, 4, H, W) - RGBA 四通道输入
        """
        # 1. 拆分输入
        x_rgb = x[:, :3, :, :]
        x_spec = x[:, 3:, :, :]

        outs = []
        # rgb_feat 初始为图像输入 (B, 3, H, W)
        rgb_feat = x_rgb
        spec_feat = x_spec
        
        # 保存各阶段的光谱特征
        spec_features = []

        # 循环处理 4 个阶段
        for i in range(self.num_stages):
            # ===========================
            # A. 光谱流 (Auxiliary Stream)
            # ===========================
            # 通过 CNN Block，输出 4D 张量
            spec_feat = self.spec_stages[i](spec_feat)
            spec_features.append(spec_feat)  # 保存当前阶段的光谱特征

            # ===========================
            # B. RGB流 (Main Stream) - 获取父类组件
            # ===========================
            # 从 self.layers 中解包当前阶段的组件
            # 标准 MiT 结构：[patch_embed, blocks, norm]
            patch_embed, blocks, norm = self.layers[i]

            # ===========================
            # C. RGB流 - Patch Embedding
            # ===========================
            # MMseg 1.x 中 patch_embed 返回 (tokens, (H, W))
            # 使用嵌套解包正确获取 H 和 W
            rgb_feat_tokens, (H, W) = patch_embed(rgb_feat)

            # ===========================
            # D. 融合 (Fusion)
            # ===========================
            # 融合模块需要 4D 张量 (B, C, H, W) 作为输入
            # 将 Transformer 的 tokens 转回 4D 张量
            rgb_feat_4d = rgb_feat_tokens.reshape(
                rgb_feat_tokens.shape[0], H, W, rgb_feat_tokens.shape[-1]).permute(0, 3, 1, 2)
            
            # 执行融合：RGB 4D + 投影后的光谱 4D
            fused_feat_4d = self.fusion_blocks[i](rgb_feat_4d, spec_feat)
            
            # 将融合后的特征转回 tokens (B, N, C) 给 Transformer Block 使用
            rgb_feat_tokens = fused_feat_4d.flatten(2).transpose(1, 2)

            # ===========================
            # E. RGB流 - Transformer Blocks & Norm
            # ===========================
            # 输入输出都是 tokens (B, N, C)
            for blk in blocks:
                # 【关键修复】MMSeg 1.x 的 TransformerBlock forward 需要 hw_shape 参数
                # 将 H 和 W 打包成元组传入
                rgb_feat_tokens = blk(rgb_feat_tokens, hw_shape=(H, W))
                
            rgb_feat_tokens = norm(rgb_feat_tokens)

            # ===========================
            # F. 输出准备
            # ===========================
            # 将最终 tokens 转回 4D 张量，作为本阶段输出，以及下一阶段的输入
            rgb_feat_out = rgb_feat_tokens.reshape(
                rgb_feat_tokens.shape[0], H, W, rgb_feat_tokens.shape[-1]).permute(0, 3, 1, 2)
            
            outs.append(rgb_feat_out)
            # 重要：更新 rgb_feat 为当前阶段输出的 4D 张量，用于下一阶段的 Patch Embed 输入
            rgb_feat = rgb_feat_out 
        
        # ===========================
        # G. 跨阶段特征融合（新增）
        # ===========================
        import torch.nn.functional as F
        
        # 示例：将第4阶段的光谱特征上采样到第3阶段的尺寸，并与第3阶段的RGB特征融合
        if len(spec_features) >= 4 and len(outs) >= 3:
            spec_feat_s4 = spec_features[3]  # 第4阶段的光谱特征
            rgb_feat_s3 = outs[2]  # 第3阶段的RGB特征
            
            # 上采样光谱特征到RGB特征的尺寸
            spec_feat_upsampled = F.interpolate(
                spec_feat_s4, 
                size=rgb_feat_s3.shape[2:],  # 目标 H, W
                mode='bilinear',  # 双线性插值，比较平滑
                align_corners=False
            )
            
            # 可以选择添加额外的融合操作
            # 例如：创建新的融合模块或直接相加
            # fused_feature = self.fusion_blocks[2](rgb_feat_s3, spec_feat_upsampled)
            # outs[2] = fused_feature  # 更新第3阶段的输出

        return outs