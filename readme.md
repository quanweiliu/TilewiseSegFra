# Multimodal semantic segmentation

# Summary
| Code      | Paper |  Journal |  Year | 
| ----------- | ----------- |----------- |----------- |
||**Single modality models**||
| [FDGC](https://github.com/quanweiliu/FDGC)      | [A Fast Dynamic Graph Convolutional Network and CNN Parallel Network for Hyperspectral Image Classification](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/abstract/document/9785802)       | TGRS       | 2022       | 
||**Multimodal models**||
| [RDFNet](https://github.com/quanweiliu/TilewiseSegFra)      | [RDFNet: RGB-D Multi-level Residual Feature Fusion for Indoor Semantic Segmentation](https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_RDFNet_RGB-D_Multi-Level_ICCV_2017_paper.pdf)       | TGRSL       | 2017       |


# Description
This repository proposed a new taxonomy to descibe existed tile based image semantic segmenation models.

Based the input and output of the model, we categorized these pixel-level classification model into singlesacle singlemodality input and singleoutput (SSISO), singlesacle multimodelity input and singleoutput (SMISO), singlesacle multimodelity input and multioutput (SMIMO), mutlisacle multimodelity input and singleoutput (MMISO), mutlisacle multimodelity input and multiouput (MMIMO).

Of course, there are multiscale singlemodality input, singleoutput (MSISO) and multiscale singlemodality input and multioutput (MSIMO) and so on. We will continue and add them in this framework.


# Nomalization:
1. MinMax normalization
$$
x_i' = \frac{x_i - \min(x)}{\max(x) - \min(x)}
$$

**特点**：
- 操作单位：每个通道（维度）
- 输出范围固定（如 [0, 1]）
- 保留比例信息（但不保留方向信息）
- 对极值敏感

2. Unit-norm normalization
$$
\mathbf{x'} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} = \frac{\mathbf{x}}{\sqrt{\sum_i x_i^2}}
$$

**特点**：
- 操作单位：每个样本/像素向量
- 所有向量模长为 1，分量值通常在 [-1, 1] 范围
- 强调方向信息（如光谱形状）
- 不保留幅值大小
> 在高光谱图像处理任务中，单位范数归一化更适合注重“光谱形状”的任务；而 Min-Max 更适合需要统一数值尺度或可视化展示的场景.但是要明确，**使用单位范数归一化的时候，不要使用降维**。

# Tricks
- 添加环境变量

```pyhon
import sys
sys.path.append('/content/drive/MyDrive/code/MDL/')
```
- 更改所用的 GPU
```pyhon
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [0]))
```


Noting: 
- 配置文件在“extraction_epoch.yml”
- 更换数据集后，需要在dataloader文件中修改数据的mean和std
- dataLoader 构造训练和测试数据集。
- dataLoader2 专为高光谱图像构造双分支输入的数据集。



I have collected a range of models based this taxonomy. If you want to contribute this repository and make it better, feel free to contact me. My emial : quanwei.liu@my.jcu.edu.au




