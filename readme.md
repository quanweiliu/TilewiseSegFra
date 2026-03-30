# Multimodal semantic segmentation

## Summary
| Code     |Backbone  | Paper |  Journal |  Year | 
| ----------- | ----------- | ----------- |----------- |----------- |
||**Single modality models**||
| [ABCNet](https://github.com/lironui/ABCNet)     | ResNet | [ABCNet: Attentive Bilateral Contextual Network for Efficient Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0924271621002379)       | ISPRS       | 2021       | 
| [AMSUnet](https://github.com/llluochen/AMSUnet)     |U-Net-like   | [AMSUnet: A Neural Network using Atrous Multi-scale Convolution for Medical Image Segmentation](https://www.sciencedirect.com/science/article/pii/S0010482523005851)       | Computers in Biology and Medicine       | 2023    | 
| [SAM](https://github.com/facebookresearch/segment-anything)    | ViT   | [Segment Anything](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html)       | CVPR       | 2023       |
| [SAM_MLoRA](https://github.com/xiaoyan07/SAM_MLoRA/blob/main/networks/sam_multi_lora.py)     |SAM   | [Multi-LoRA Fine-Tuned Segment Anything Model for Urban Man-Made Object Extraction](https://ieeexplore.ieee.org/abstract/document/10637992)       | TGRS       | 2024       |
||**Multimodal models**||
| [ACNet](https://github.com/anheidelonghu/ACNet)    | ResNet    | [ACNET: Attention Based Network to Exploit Complementary Features for RGBD Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/8803025)       | ICIP       | 2019       |
| [MCANet](https://github.com/yisun98/SOLC/tree/main)     |U-Net-like   | [MCANet: A joint Semantic Segmentation Framework of Optical and SAR Images for Land Use Classification](https://www.sciencedirect.com/science/article/pii/S0303243421003457)       | JAG       | 2021       |
| [SOLC](https://github.com/yisun98/SOLC/tree/main)    |U-Net-like    | [Remote Sensing Sar-Optical Land-use Classfication](https://github.com/yisun98/SOLC/tree/main)       | GitHub       | 2022       |
| FAFNet      |ResNet  | [FAFNet: Fully Aligned Fusion Network for RGBD Semantic Segmentation Based on Hierarchical Semantic Flows](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/ipr2.12614)       | IET IP       | 2022       |
| [CMFNet](https://github.com/sstary/SSRS/tree/main/CMFNet)    |SegNet    | [A Crossmodal Multiscale Fusion Network for Semantic Segmentation of Remote Sensing Data](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/ipr2.12614)       | JSTAR       | 2022       |
| CMANet     |ResNet   | [CMANet: Cross-Modality Attention Network for Indoor-Scene Semantic Segmentation](https://www.mdpi.com/1424-8220/22/21/8520)       | Sensors      | 2022       |
| CANet     |ResNet   | [CANet: Co-attention network for RGB-D semantic segmentation](https://www.sciencedirect.com/science/article/pii/S0031320321006440)       | PR       | 2022       |
| CMGFNet     |ResNet   | [CMGFNet: A Deep Cross-modal Gated Fusion Network for Building Extraction from Very High-resolution Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0924271621003294)       | ISPRS       | 2022       |
| [PCGNet](https://github.com/hmdliu/PCGNet)    |ResNet    | [Pyramid-Context Guided Feature Fusion for RGB-D Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/9859353)       | ICMEW       | 2022       |
| [DE-DCGCN-EE](https://github.com/YangLibuaa/DE-DCGCN-EE)    | U-Net-like   | [Dual Encoder-Based Dynamic-Channel Graph Convolutional Network With Edge Enhancement for Retinal Vessel Segmentation](https://ieeexplore.ieee.org/abstract/document/9714302)       | TMI       | 2022       |
| [SFAF-MA](https://github.com/hexunjie/SFAF-MA)    | ResNet  | [SFAF-MA: Spatial Feature Aggregation and Fusion With Modality Adaptation for RGB-Thermal Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/10103760)       | CVPR       | 2023       |
| [PACSCNet](https://github.com/F8AoMn/PACSCNet)     |Res2Net   | [Progressive Adjacent-Layer Coordination Symmetric Cascade Network for Semantic Segmentation of Multimodal Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0957417423025010)       | ESA       | 2023       |
| DECCFNet     | ResNet   | [A Deep Cross-Modal Fusion Network for Road Extraction With High-Resolution Imagery and LiDAR Data](https://ieeexplore.ieee.org/document/10439005)       | TGRS       | 2024       |
| [AsymFormer](https://github.com/Fourier7754/AsymFormer)    |MiT + ConvNeXt   | [AsymFormer: Asymmetrical Cross-Modal Representation Learning for Mobile Platform Real-Time RGB-D Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024W/USM/papers/Du_AsymFormer_Asymmetrical_Cross-Modal_Representation_Learning_for_Mobile_Platform_Real-Time_RGB-D_CVPRW_2024_paper.pdf)       | TIM       | 2023       |
| [MGFNet](https://github.com/yeyuanxin110/YESeg-OPT-SAR)   | ResNet + HRNet | [MGFNet: An MLP-dominated Gated Fusion Network for Semantic Segmentation of High-resolution Multi-modal Remote Sensing images](https://www.sciencedirect.com/science/article/pii/S1569843224005971)       | JAG       | 2024       |
| [MGFNet](https://github.com/DrWuHonglin/MGFNet)    | ResNet   | [MGFNet: a Multiscale Gated Fusion Network for Multimodal Semantic Segmentation](https://link.springer.com/article/10.1007/s00371-025-03912-x)       | The Visual Computer       | 2025       |

## Description
This repository proposes a new taxonomy to descibe existed tile based image semantic segmenation models.

Based the input and output of the model, we categorized these pixel-level classification model into singlesacle singlemodality input and singleoutput (SSISO), singlesacle multimodelity input and singleoutput (SMISO), singlesacle multimodelity input and multioutput (SMIMO), mutlisacle multimodelity input and singleoutput (MMISO), mutlisacle multimodelity input and multiouput (MMIMO).

Of course, there are multiscale singlemodality input, singleoutput (MSISO) and multiscale singlemodality input and multioutput (MSIMO) and so on. We will continue and add them in this framework.


Distributed Training: The training process employs distributed training, while inference is validated solely on a single GPU to simplify the code.


If you want to contribute this repository and make it better, feel free to contact me. My emial : quanwei.liu@my.jcu.edu.au


## Files

ISPRS_loader, ISPRS_loader2 和 ISPRS_loader3 三个文件都是用于加载数据。

- ISPRS_loader 是通过 train / val / test 文件夹加载数据。
- ISPRS_loader2 是通过 train.txt / val.txt / test.txt 文件加载数据。
- ISPRS_loader3 融合 ISPRS_loader 和 ISPRS_loader2，得到一个加载速度快的 train.txt / val.txt / test.txt 文件加载数据。

ISPRS_loader 和 ISPRS_loader2 两种数据加载方式的精度是差不多一致的，但是 ISPRS_loader2 采用了更强的数据增强，复杂的数据增强过程导致模型训练非常慢，并且需要更长的训练周期。去掉多尺度等复杂的变化过程，会极大的加快训练过程。因此，我结合 ISPRS_loader 和 ISPRS_loader2，构建一个基于 train.txt / val.txt / test.txt 加载的data loader。

- OSTD_loader 和 OSTD_loader2 效果一摸一样。我用来测试norm 位置，发现，数据增强如果不改变数据的分布的话，放哪都一样。

因为得到的精度一致，ISPRS_loader2加载会显著拖慢训练时间，所以以后都用 ISPRS_loader, ISPRS_loader3 文件加载数据。


## Nomalization:
1. MinMax normalization

$$
x_i' = \frac{x_i - \min(x)}{\max(x) - \min(x)}
$$

**特点**：
- 操作单位：每个通道（维度）
- 输出范围固定（如 [0, 1]）
- 保留比例信息（但不保留方向信息）
- 对极值敏感
- 常用于 图像像素值（0–255 → 0–1）、激光雷达反射强度、NDVI 等指标。

2. Unit-norm normalization

$$
\mathbf{x'} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} = \frac{\mathbf{x}}{\sqrt{\sum_i x_i^2}}
$$

**特点**：
- 操作单位：每个样本/像素向量
- 所有向量模长为 1，分量值通常在 [-1, 1] 范围
- 强调方向信息（如光谱形状）
- 不保留幅值大小


3. Z-score standardization

$$
\hat{x} = \frac{x - \mu}{\sigma + \varepsilon}
$$

- x 表示原始输入向量或矩阵；
- 𝜇=E[x] 表示均值；
- σ=Var[x]表示标准差；
- ε 是防止分母为零的微小常数。
- 保留了数据的相对差异（梯度、对比度信息）。
- 常用于 深度 / 高光谱 / 特征图 等本身数值分布较大、需要标准化的输入。




> 在高光谱图像处理任务中，单位范数归一化更适合注重“光谱形状”的任务；而 Min-Max 更适合需要统一数值尺度或可视化展示的场景.但是要明确，**使用单位范数归一化的时候，不要使用降维**。

## Tricks
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
- 训练调试过程中发现，PACSCNet learning rate 设置大一点，就不能训练了，好敏感这个模型。

## pretrains

The segformer pretrains for AsymFormer comes from  "Official PyTorch implementation of SegFormer" https://github.com/NVlabs/SegFormer?tab=readme-ov-file.








## Reference

You may want to cite:

```
@article{liu2025pixels,
  title={From Pixels to Images: Deep Learning Advances in Remote Sensing Image Semantic Segmentation},
  author={Liu, Quanwei and Huang, Tao and Dong, Yanni and Yang, Jiaqi and Xiang, Wei},
  journal={arXiv preprint arXiv:2505.15147},
  year={2025}
}
```



### License

Code in this repo is for non-commercial use only.


### Acknowledge


[PatchwiseClsFra](https://github.com/quanweiliu/PatchwiseClsFra)
[finetune-anything](https://github.com/ziqi-jin/finetune-anything/tree/main)