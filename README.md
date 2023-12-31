# 基于监控的天气识别系统

![SWJTU Logo](https://www.swjtu.edu.cn/images/logo.png)

## 介绍

这是一个使用PyTorch库进行深度学习模型训练的Python脚本。

### 分类

| 代码 | 天气     | target |
| ---- | -------- | ------ |
| q    | 晴       | 0      |
| e    | 阴       | 1      |
| r    | 雨       | 2      |
| a    | 雪       | 3      |
| s    | 雾       | 4      |
| f    | 夜晚     | 5      |
| d    | 应该删除 | 7      |

### 数据集

所用数据不在github上，须另行下载

## 文件说明

- `train.py`: 主训练脚本。
- `utils.py`: 包含一些实用函数，如绘制曲线和删除文件夹内容。
- `dataset.py`: 包含数据集类。
- `settings.py`: 包含所有的设置和配置。

## 安装

要安装此项目，你需要首先克隆或下载此仓库到你的本地机器。

接下来，你需要安装项目的依赖。这个项目的所有Python依赖都列在`requirements.txt`文件中。你可以使用以下命令安装这些依赖：

```bash
pip install -r requirements.txt
```

这个命令将会安装`requirements.txt`文件中列出的所有包。

## 依赖

- PyTorch
- torchvision
- loguru

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
