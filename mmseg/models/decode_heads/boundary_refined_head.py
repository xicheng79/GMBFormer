import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize

@MODELS.register_module()
class BoundaryRefinedHead(BaseDecodeHead):
    """
    Boundary-Aware Feature Fusion Head for High-Resolution Green Space Extraction.
    针对城市绿地提取设计的边界感知解码器
    """
    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        self.feature_strides = feature_strides
        
        # 1. 统一通道数的层 (Embedding Layers)
        embedding_dim = self.channels
        self.linear_layers = nn.ModuleList([
            ConvModule(
                in_channels=in_c,
                out_channels=embedding_dim,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            for in_c in self.in_channels
        ])

        # 2. 边界细化模块 (针对 Stage 1, 也就是 scale 1/4 的特征)
        # 使用 3x3 卷积提取局部细节
        self.detail_refine = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1,
            groups=embedding_dim, # 深度可分离卷积，减少参数
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        # 3. 空间注意力融合模块 (Spatial Attention Fusion)
        # 将低层特征作为 Gate 来加权高层特征
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(embedding_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 4. 最终融合层
        self.fusion_conv = ConvModule(
            in_channels=embedding_dim * 4, # 4个层级拼接
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # inputs 是一个列表，包含 [c1, c2, c3, c4]
        # c1: 1/4, c2: 1/8, c3: 1/16, c4: 1/32
        
        inputs = self._transform_inputs(inputs)
        outs = []

        # -----------------------------------------------
        # Step 1: 统一通道数
        # -----------------------------------------------
        processed_inputs = []
        for idx, layer in enumerate(self.linear_layers):
            processed_inputs.append(layer(inputs[idx]))
        
        c1, c2, c3, c4 = processed_inputs

        # -----------------------------------------------
        # Step 2: 边界增强机制 (核心创新点)
        # -----------------------------------------------
        # 对最浅层的 C1 (包含最丰富的纹理/边界) 进行增强
        c1_detail = self.detail_refine(c1)
        
        # 计算空间注意力图 (Attention Map)
        # 意思：如果 C1 中某个位置激活值很高（通常是边缘或纹理丰富处），
        # 我们就赋予该位置更高的权重
        spatial_att = self.spatial_gate(c1_detail) 

        # -----------------------------------------------
        # Step 3: 多尺度特征融合
        # -----------------------------------------------
        upsampled_outs = [c1_detail] # 放入增强后的 C1
        
        # 处理 C2, C3, C4
        for i, feat in enumerate([c2, c3, c4]):
            # 上采样到 C1 的尺寸 (1/4)
            feat_up = resize(
                input=feat,
                size=c1.shape[2:],
                mode='bilinear',
                align_corners=False)
            
            # 关键点：用 C1 生成的 Attention 来加权深层特征
            # 这样可以防止深层特征的模糊边界“污染”了浅层的清晰边界
            # 公式：F_refined = F_up * (1 + Attention)
            feat_refined = feat_up * (1 + spatial_att)
            
            upsampled_outs.append(feat_refined)

        # -----------------------------------------------
        # Step 4: 拼接与预测
        # -----------------------------------------------
        concat_feats = torch.cat(upsampled_outs, dim=1)
        fusion_feats = self.fusion_conv(concat_feats)
        
        out = self.cls_seg(fusion_feats)
        
        return out