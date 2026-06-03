# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmseg.models.utils import resize
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS

@MODELS.register_module()
class PanoOptiNet_EncoderDecoder(EncoderDecoder):
    """基于继承的PanoOptiNet滑窗机制分割器，支持传递overlap信息。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap = None

    def extract_feat(self, img, img_metas=None, overlap=None):
        """从图像中提取特征，增加overlap参数用于特殊处理。"""
        # img could be a list of tensors in mmcv/mmseg 1.x pipeline during inference due to Some pack operations, unwrap it
        if isinstance(img, list):
            img = img[0]
            
        # mmseg 1.x extract_feat args: import tensor img
        if img_metas is None:
             x = self.backbone(img)
        else:
             x = self.backbone(img, img_metas, overlap)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs, batch_img_metas):
        """编码解码流程，加入overlap参数用于引导窗口。"""
        x = self.extract_feat(inputs, batch_img_metas, self.overlap)
        out = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        
        # NOTE: PanoOptiNetHead.predict/forward_test returns a tuple (seg_logits, overlap) instead of just seg_logits
        if isinstance(out, tuple):
            out = out[0]
            
        out = resize(
            input=out,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def predict(self, inputs, data_samples=None):
        """预测接口，适配 mmseg 1.x。"""
        # img could miss batch dimension during inference due to LoadImageFromNDArray directly passing C,H,W, unsqueeze it
        if isinstance(inputs, list):
            inputs = inputs[0]
            
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
            
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                scale_factor=1.0,
                filename='dummy_0_0_0'
            )]
            
        # PanoOptiNet backbone 强制需要读取 filename 按照特定格式解析 dx, dy, id，所以补充假数据防止报错
        for meta in batch_img_metas:
            if 'filename' not in meta:
                meta['filename'] = 'dummy_0_0_0'
        
        seg_logits = self.encode_decode(inputs, batch_img_metas)
        
        return self.postprocess_result(seg_logits, data_samples)

    def _decode_head_forward_train(self, inputs, data_samples):
        # NOT IMPLEMENTED FOR 1x predict
        pass
    
    def loss(self, inputs, data_samples):
        # NOT IMPLEMENTED FOR 1x predict
        pass
