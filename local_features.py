import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    通过全局平均池化提取通道统计信息，再学习每个通道的重要性权重，
    从而增强有效特征、抑制无关特征。
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # squeeze：把空间维度压缩成 1x1 的通道描述符
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # excitation：两层全连接学习通道之间的依赖关系
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b: batch size, c: channels
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        # 将通道权重逐元素作用到原特征图上
        return x * y


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：Depthwise Conv + Pointwise Conv。

    先按通道做空间卷积，再用 1x1 卷积融合通道，参数量和计算量都更小。
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise：每个输入通道单独做 3x3 卷积
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False
        )
        self.bn_depth = nn.BatchNorm2d(in_ch)
        # Pointwise：用 1x1 卷积完成通道混合并调整通道数
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_point = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_depth(self.depthwise(x)))
        x = self.relu(self.bn_point(self.pointwise(x)))
        return x


class LocalFeatures(nn.Module):
    """LSEVGG：轻量级 SE-VGG 风格网络。

    结构上保留了 VGG 的分阶段堆叠方式，同时用深度可分离卷积降低开销，
    再通过 SE 模块增强通道表达能力。num_classes 可按数据集配置。
    """
    def __init__(self, out_channels=256):
        super().__init__()

        # Block 1：先用标准卷积建立浅层特征，再进入轻量卷积和注意力模块
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2, 2)
        )

        # Block 2：逐步提升通道数，扩大表征能力
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d(2, 2)
        )

        # Block 3：继续加深网络，让模型捕获更复杂的局部结构
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block3(self.block2(self.block1(x)))

