## 基于监控的天气识别系统

![SWJTU Logo](https://www.swjtu.edu.cn/images/logo.png)

### 分类

| 代码 | 天气     | target |
| ---- | -------- | ------ |
| q    | 晴       | 0      |
| e    | 阴       | 1      |
| r    | 雨       | 2      |
| a    | 雪       | 3     |
| s    | 雾       | 4      |
| f    | 夜晚     | 5      |
| d    | 应该删除 | 7      |

### WARRING

所用数据不在github上，须另行下载

### 介绍

极早期开发阶段，完全是萌新初学的程度

GitHub Copilot: # 深度学习模型训练脚本

这是一个使用PyTorch库进行深度学习模型训练的Python脚本。

## 依赖

- PyTorch
- torchvision
- loguru

## 文件说明

- `train.py`: 主训练脚本。
- `utils.py`: 包含一些实用函数，如绘制曲线和删除文件夹内容。
- `dataset.py`: 包含数据集类。
- `settings.py`: 包含所有的设置和配置。

## 使用方法

1. 安装所有依赖：`pip install -r requirements.txt`
2. 运行训练脚本：`python train.py`

## 功能

- 数据加载和预处理
- 模型创建
- 模型训练和验证
- 模型保存和加载
- 训练进度可视化

## 注意事项

- 请确保你的数据集在正确的路径下。
- 你可以在`settings.py`中修改训练参数和配置。
- 如果你想使用自己的模型，你可以在`train.py`中修改模型创建部分的代码。

## 贡献

欢迎任何形式的贡献，包括但不限于问题报告、功能请求、文档改进和代码提交。
