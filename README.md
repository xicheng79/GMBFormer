# GMBFormer

> A MMSegmentation-based implementation of an NDVI-guided global memory bank for urban green-space segmentation.

[中文说明](README.zh-CN.md)

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c">
  <img alt="MMSegmentation" src="https://img.shields.io/badge/MMSegmentation-1.x-1677ff">
  <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-green">
</p>

GMBFormer is a codebase for training and evaluating an urban green-space segmentation model with RGB images and an NDVI channel. It is built on MMSegmentation 1.x and keeps the standard OpenMMLab workflow for config-driven training, testing, logging, and visualization.

## What Is Included

- `GMBEncoderDecoder`: separates RGB and NDVI from RGBA-style inputs.
- `SegformerGMBHead`: adds a global memory bank branch to a SegFormer-style decode head.
- `GlobalMemoryBankModule`: writes vegetation-rich features into memory and reads them through cross-attention.
- `LoadRGBAImageFromFile`: loads RGB plus NDVI images as four-channel inputs.
- `PhotoMetricDistortionRGBA`: applies photometric augmentation to RGB while preserving NDVI.
- `GMBFormer/config.py`: the main training and evaluation config.

## Repository Layout

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

The custom model code is mainly in `GMBFormer/`. The custom RGBA data loading and augmentation code is in `mmseg/datasets/transforms/rgba_transforms.py`.

## Environment

Create an environment and install dependencies:

```bash
conda create -n gmbformer python=3.8 -y
conda activate gmbformer
```

Install PyTorch according to your CUDA version:

```bash
# See https://pytorch.org/get-started/locally/
```

Install OpenMMLab dependencies and this repository:

```bash
pip install -U openmim
mim install mmengine mmcv
pip install -r requirements.txt
pip install -v -e .
```

## Data Format

The default config uses VOC-style segmentation data. Each image is expected to be an RGBA PNG:

- `R/G/B`: image channels used by the backbone.
- `A`: normalized NDVI channel used by the memory bank.
- mask: PNG segmentation mask. The ignore label is `255`.

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

Update these fields in `GMBFormer/config.py` before training:

- `data_root`
- `num_classes`
- `metainfo.classes`
- `metainfo.palette`
- `work_dir`

## Train

```bash
python tools/train.py GMBFormer/config.py
```

The config imports the local custom modules through:

```python
custom_imports = dict(
    imports=['GMBFormer', 'mmseg.datasets.transforms.rgba_transforms'],
    allow_failed_imports=False
)
```

Training outputs are saved to the directory specified by `work_dir` in `GMBFormer/config.py`.

## Evaluate

Use the same config and replace the checkpoint path with your actual model file:

```bash
python tools/test.py GMBFormer/config.py work_dirs/custom_gmb/your_checkpoint.pth
```

Save visualized predictions:

```bash
python tools/test.py GMBFormer/config.py work_dirs/custom_gmb/your_checkpoint.pth --show-dir work_dirs/custom_gmb/vis
```

## Notes

- The backbone receives RGB by default, while NDVI is kept as a separate memory-admission signal.
- If your input images do not contain NDVI in the alpha channel, update the data pipeline and `use_ndvi_channel` settings.
- If you change class definitions, update both the model head and dataset `metainfo`.

## License

This repository follows the Apache-2.0 license inherited from MMSegmentation unless otherwise specified.

## Contact

- Xi Cheng: `chengxi13@cdut.edu.cn`
