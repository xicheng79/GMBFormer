# GMBFormer

> 基于 MMSegmentation 的 NDVI 引导全局记忆库城市绿地分割代码实现。

[English README](README.md)

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c">
  <img alt="MMSegmentation" src="https://img.shields.io/badge/MMSegmentation-1.x-1677ff">
  <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-green">
</p>

GMBFormer 是一个用于训练和评估城市绿地分割模型的代码库。输入采用 RGB 图像和 NDVI 通道，整体流程基于 MMSegmentation 1.x，保留 OpenMMLab 的配置式训练、测试、日志和可视化方式。

## 包含内容

- `GMBEncoderDecoder`：从 RGBA 风格输入中分离 RGB 和 NDVI。
- `SegformerGMBHead`：在 SegFormer 风格解码头中加入全局记忆库分支。
- `GlobalMemoryBankModule`：将植被特征写入记忆库，并通过交叉注意力读取。
- `LoadRGBAImageFromFile`：读取 RGB 加 NDVI 的四通道图像。
- `PhotoMetricDistortionRGBA`：只增强 RGB，保留 NDVI 通道不变。
- `GMBFormer/config.py`：当前主训练和测试配置文件。

## 仓库结构

```text
GMBFormer/
|-- GMBFormer/
|   |-- config.py
|   |-- global_memory_bank.py
|   |-- segformer_gmb_head.py
|   `-- gmb_encoder_decoder.py
|-- mmseg/datasets/transforms/
|   `-- rgba_transforms.py
|-- tools/
|   |-- train.py
|   `-- test.py
|-- requirements/
`-- work_dirs/
```

核心模型代码主要在 `GMBFormer/` 中，RGBA 数据读取和增强代码在 `mmseg/datasets/transforms/rgba_transforms.py` 中。

## 环境安装

创建环境：

```bash
conda create -n gmbformer python=3.8 -y
conda activate gmbformer
```

根据自己的 CUDA 版本安装 PyTorch：

```bash
# 参考 https://pytorch.org/get-started/locally/
```

安装 OpenMMLab 依赖和本仓库：

```bash
pip install -U openmim
mim install mmengine mmcv
pip install -r requirements.txt
pip install -v -e .
```

## 数据格式

默认配置使用 VOC 风格的分割数据。图像应为 RGBA PNG：

- `R/G/B`：进入骨干网络的图像通道。
- `A`：归一化后的 NDVI 通道，用于记忆库写入判断。
- 标签：PNG 分割掩码，忽略标签为 `255`。

```text
data_root/
|-- JPEGImages/
|   |-- xxx.png
|   `-- ...
|-- SegmentationClass/
|   |-- xxx.png
|   `-- ...
`-- ImageSets/Segmentation/
    |-- train.txt
    `-- val.txt
```

训练前请在 `GMBFormer/config.py` 中修改：

- `data_root`
- `num_classes`
- `metainfo.classes`
- `metainfo.palette`
- `work_dir`

## 训练

```bash
python tools/train.py GMBFormer/config.py
```

配置文件通过下面的字段导入本仓库自定义模块：

```python
custom_imports = dict(
    imports=['GMBFormer', 'mmseg.datasets.transforms.rgba_transforms'],
    allow_failed_imports=False
)
```

训练输出目录由 `GMBFormer/config.py` 中的 `work_dir` 控制。

## 评价

测试时使用同一个配置文件，并将权重路径替换为自己的模型文件：

```bash
python tools/test.py GMBFormer/config.py work_dirs/custom_gmb/your_checkpoint.pth
```

保存预测可视化结果：

```bash
python tools/test.py GMBFormer/config.py work_dirs/custom_gmb/your_checkpoint.pth --show-dir work_dirs/custom_gmb/vis
```

## 注意事项

- 默认设置下，骨干网络只接收 RGB，NDVI 作为独立的记忆写入信号。
- 如果输入图像的 alpha 通道不是 NDVI，需要同步修改数据 pipeline 和 `use_ndvi_channel` 设置。
- 如果修改类别定义，需要同时更新模型解码头和数据集 `metainfo`。

## 许可证

除非特别说明，本仓库遵循 MMSegmentation 继承的 Apache-2.0 开源许可证。

## 联系

- Xi Cheng: `chengxi13@cdut.edu.cn`
