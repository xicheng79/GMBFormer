"""
global_memory_bank.py
=====================
全局动态记忆库模块 (Global Dynamic Memory Bank Module)

设计思路：
  - 一张 3968×3968 图像被切为 100 个 512×512 Patch
  - 这 100 个 Patch 共享同一个全局记忆库
  - NDVI 均值高（绿地丰富）的 Patch → 写入记忆库 (FTC重构)
  - 所有 Patch 通过 Cross-Attention 从记忆库读取增强特征 (FTP重构)
  - 记忆库用 EMA（指数移动平均）动量更新，保证平滑稳定
  - 非绿地 Patch（NDVI 低）不写入，避免负迁移

NDVI 存储约定：
  - PNG 文件的 Alpha 通道（第 4 通道）
  - 已归一化到 [0, 1]，读进来就是 float 形式
  - 不需要额外缩放
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMemoryBankModule(nn.Module):
    """全局动态绿地特征记忆库

    核心角色：
        FTC 重构 (write): NDVI 均值 > ndvi_thresh → 提取纯净绿地特征 → EMA 更新记忆库
        FTP 重构 (read) : 当前 Patch 特征作 Query → Cross-Attention 检索记忆库 → 融合增强

    Args:
        in_channels (int): 输入特征图通道数（建议接 MiT-B4 Stage-4 的 512）
        memory_size (int): 记忆库槽位数，维护多少种不同绿地原型，默认 64
        momentum (float): EMA 更新动量，越大记忆更新越慢越稳定，默认 0.99
        ndvi_thresh (float): NDVI 均值阈值（归一化后 [0,1] 尺度），超过此值才写入记忆库。
                             默认 0.6，对应原始 NDVI ≈ 0.2（中等植被覆盖）。
                             换算关系：normalized = (NDVI_raw + 1) / 2
        num_heads (int): Cross-Attention 多头数，默认 8
    """

    def __init__(self,
                 in_channels: int,
                 memory_size: int = 64,
                 momentum: float = 0.99,
                 ndvi_thresh: float = 0.6,
                 num_heads: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.memory_size = memory_size
        self.momentum = momentum
        self.ndvi_thresh = ndvi_thresh
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, \
            f"in_channels({in_channels}) 必须能被 num_heads({num_heads}) 整除"

        # ── 记忆库 ──────────────────────────────────────────────────────────
        # shape: [M, C]，L2 归一化存储，不参与梯度回传
        self.register_buffer(
            'memory_bank',
            F.normalize(torch.randn(memory_size, in_channels), p=2, dim=1)
        )

        # ── Cross-Attention 投影层 ───────────────────────────────────────────
        # Q 来自当前特征图（空间维度保留），用 1×1 Conv
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        # K, V 来自记忆库（向量），用 Linear
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)

        # ── 输出融合 ─────────────────────────────────────────────────────────
        # 拼接原始特征 + 检索特征 → 压缩回 in_channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # ── 门控标量：控制记忆增强的强度（可学习）────────────────────────────
        # 初始化为 -3，sigmoid(-3)≈0.05，让 memory 分支初期贡献极小，
        # 待 backbone 建立稳定 RGB 表征后再逐渐打开
        self.gate = nn.Parameter(torch.full((1,), -3.0))

    # =========================================================================
    #   FTP 重构：从记忆库 Cross-Attention 读取特征
    # =========================================================================
    def read_memory(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] 当前 Patch 的深层特征图

        Returns:
            enhanced: [B, C, H, W] 记忆增强后的特征图
        """
        B, C, H, W = features.shape
        M = self.memory_size

        # Q: [B, C, H*W] → [B, H*W, num_heads, head_dim]
        q = self.q_proj(features)                              # [B, C, H, W]
        q = q.view(B, self.num_heads, self.head_dim, H * W)   # [B, nh, hd, HW]
        q = q.permute(0, 3, 1, 2)                             # [B, HW, nh, hd]

        # K, V from memory bank: [M, C]
        k = self.k_proj(self.memory_bank)                     # [M, C]
        v = self.v_proj(self.memory_bank)                     # [M, C]
        k = k.view(M, self.num_heads, self.head_dim)          # [M, nh, hd]
        v = v.view(M, self.num_heads, self.head_dim)          # [M, nh, hd]

        # Attention: [B, HW, nh, hd] × [M, nh, hd] → [B, HW, nh, M]
        scale = self.head_dim ** -0.5
        # einsum: b p h d, m h d -> b p h m
        attn = torch.einsum('bphe, mhe -> bphm', q, k) * scale
        attn = F.softmax(attn, dim=-1)                        # [B, HW, nh, M]

        # Aggregate: [B, HW, nh, M] × [M, nh, hd] → [B, HW, nh, hd]
        retrieved = torch.einsum('bphm, mhd -> bphd', attn, v)  # [B, HW, nh, hd]
        retrieved = retrieved.permute(0, 2, 3, 1)               # [B, nh, hd, HW]
        retrieved = retrieved.reshape(B, C, H, W)               # [B, C, H, W]

        # 门控融合：gate 从 0 开始学习，让模型逐渐打开记忆增强
        gate = torch.sigmoid(self.gate)
        fused = torch.cat([features, retrieved * gate], dim=1)  # [B, 2C, H, W]
        enhanced = self.out_proj(fused)                         # [B, C, H, W]

        return enhanced

    # =========================================================================
    #   FTC 重构：用 NDVI 引导，将绿地特征写入记忆库（EMA 更新）
    # =========================================================================
    @torch.no_grad()
    def update_memory_with_ndvi(self,
                                 features: torch.Tensor,
                                 ndvi_maps: torch.Tensor) -> None:
        """
        用 NDVI 均值作为自监督信号，引导哪些 Patch 的特征写入记忆库。

        Args:
            features  : [B, C, H, W] 当前 batch 的**原始**深层特征（更新前）
            ndvi_maps : [B, 1, H_orig, W_orig] 或 [B, 1, H, W]，NDVI 图（已归一化 0~1）
        """
        B, C, H, W = features.shape

        for i in range(B):
            # 计算当前 Patch 的 NDVI 均值（作为绿地丰富度代理指标）
            ndvi_mean = ndvi_maps[i].mean().item()

            # NDVI 低于阈值：不是绿地主导 Patch → 跳过，避免负迁移
            if ndvi_mean < self.ndvi_thresh:
                continue

            # 将特征图做全局平均池化，得到当前 Patch 的整体语义向量 [C]
            patch_feat = features[i].mean(dim=[-2, -1])  # [C]
            patch_feat = F.normalize(patch_feat, p=2, dim=0)  # 归一化

            # 在记忆库中找余弦相似度最高的槽位（最匹配的绿地原型）
            sim = torch.mv(self.memory_bank, patch_feat)  # [M]
            best_slot = sim.argmax().item()

            # EMA 动量更新选中的槽位
            updated = (self.momentum * self.memory_bank[best_slot]
                       + (1.0 - self.momentum) * patch_feat)
            self.memory_bank[best_slot] = F.normalize(updated, p=2, dim=0)

    # =========================================================================
    #   forward：训练时 read + write，推理时只 read
    # =========================================================================
    def forward(self,
                features: torch.Tensor,
                ndvi_maps: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features : [B, C, H, W] MiT-B4 Stage-4 输出特征
            ndvi_maps: [B, 1, *, *] NDVI 图（仅训练时需要）

        Returns:
            enhanced : [B, C, H, W] 记忆增强后的特征
        """
        # 训练时：先用当前原始特征更新记忆库，再读取增强
        # 注意顺序：先写（用干净的当前特征），再读（已含最新记忆）
        if self.training and ndvi_maps is not None:
            self.update_memory_with_ndvi(features, ndvi_maps)

        # 从记忆库读取增强（训练和推理都执行）
        enhanced = self.read_memory(features)
        return enhanced
