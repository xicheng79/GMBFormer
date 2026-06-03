# README 拆分与代码库整理计划

状态：已完成

## 目标

将仓库 README 整理为两个独立版本：

- `README.md`：英文版，作为 GitHub 默认展示页面，聚焦代码使用。
- `README.zh-CN.md`：中文版，供中文读者查看，聚焦环境、数据、训练和评价。

同时检查并整理 GMBFormer 代码库中与自定义模型、配置、文档路径相关的不一致问题。

## 已观察到的问题

- 当前 `README.md` 为中英混写，不利于 GitHub 首页展示。
- README 中存在历史路径写法，但当前实际配置位于 `GMBFormer/config.py`。
- README 中存在历史包名写法，但当前实际自定义包名是 `GMBFormer`。
- 代码库保留了完整 MMSegmentation 主体，整理时应优先保证兼容性，不建议贸然移动 `mmseg/` 主体目录。

## 实施步骤

- [x] 确认 README 拆分方案与代码库整理范围。
- [x] 生成英文版 `README.md`，只保留英文内容，并在顶部提供中文版本链接。
- [x] 生成中文版 `README.zh-CN.md`，使用自然中文重写说明。
- [x] 统一 README 中的配置路径为 `GMBFormer/config.py`。
- [x] 统一 README 中的自定义包名为 `GMBFormer`。
- [x] 检查核心自定义目录 `GMBFormer/` 与配置文件命名一致性。
- [x] 删除 README 中的图片展示、结果表和 Citation。
- [x] 删除 README 中与论文发布状态相关的表述。
- [ ] 检查仓库根目录中非核心文件，列出可整理建议，不直接删除。

## 可选整理项

- [x] 保留根目录 `README.zh-CN.md`，方便 GitHub 首页直接跳转。
- [ ] 将实验草稿、论文图片、临时日志单独归档到 `assets/`、`paper/` 或 `archive/`。
- [ ] 清理 `__pycache__/`、历史日志和临时脚本，但需确认后再执行。
- [ ] 将 GMBFormer 自定义模块进一步拆为 `GMBFormer/models/`、`GMBFormer/datasets/`、`GMBFormer/configs/`，但这会影响导入路径，建议作为第二阶段。

## 评审记录

- 已按代码项目主页风格重写英文 README，去掉论文式叙述。
- 已新增中文 README，并在英文 README 顶部提供跳转链接。
- 已将 README 中的核心路径说明修正为 `GMBFormer/`。
- 已删除 README 中的图片、实验结果表、Citation 和论文发布相关内容。
- 本轮未移动核心代码目录，未删除文件，未运行训练或测试。
