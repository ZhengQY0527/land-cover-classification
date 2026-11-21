# EuroSAT 地物分类：LSEVGG 与视觉 Transformer

比较局部纹理、全局布局及其融合对相似地物分类的作用。本仓库包含九种网络定义、八组完整的种子 42 实验、区域遮挡实验和轻量化评估。[29 页中文报告（含网络结构、混淆矩阵、典型错例、响应图及遮挡验证）](reports/model-report-zh.pdf)

| 组 | 模型设计 | 参数量 | 测试准确率 |
|---|---|---:|---:|
| A | 基础 ViT，4 层、128 维 | 564,490 | 95.16% |
| B | 三阶段 LSEVGG + 全局平均池化 | 254,246 | 97.80% |
| C | LSEVGG 与原图 patch 逐位置相加 + 4 层 ViT | 817,190 | 97.78% |
| D | C 移除显式位置编码 | 808,998 | 97.53% |
| E | C 将局部卷积网格平均后广播 | 817,190 | 97.51% |
| F | C 的两层 ViT 版本 | 552,230 | 97.51% |
| G | B 的第三阶段缩到 192 通道 | 174,690 | 97.60% |
| H | 192 通道卷积 + 两层 96 维 ViT | 见报告 | 试跑中止 |
| I | 卷积制块 + 三层 96 维 ViT | 246,958 | 95.48% |

训练/验证/测试为 18,900 / 4,050 / 4,050 张，按类别分层、图像级划分。各组按验证宏 F1 选取权重。机载延迟及跨地区泛化尚待实测。

## 环境与数据

实验环境为 Windows、Python 3.11、PyTorch 2.3.0+cu121 和 RTX 4060 Laptop GPU。训练和分类评估支持 CPU，完整流程的 GPU 延迟对照需要 CUDA。建议在单独的虚拟环境运行：

    python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -r requirements.txt

从 [EuroSAT 作者发布页](https://zenodo.org/records/7711810) 下载 EuroSAT_RGB.zip，解压至 data/EuroSAT_RGB/，其下是 AnnualCrop、Forest 等十个类别目录。原压缩包 MD5 为 f46e308c4d50d4bf32fedad2d3d62f3b。代码运行时现场生成权重；仓库随附可审阅的实验结果。

## 运行实验

在仓库根目录运行。完整流程耗时较长，包括 8 次各 100 轮的训练；输出写入 results/。预先整理的报告和指标在 reports/，可直接查看。

    python reproduce.py

仅运行 A/C 对照：

    python experiment.py prepare
    python experiment.py train --groups A C --seeds 42
    python experiment.py analyze --groups A C --seeds 42

如需独立复现 I，在 prepare 后把 results/manifest.json 复制到 results/serial/manifest.json，再运行：

    python experiment.py train --groups I --seeds 42 --out results/serial

相同组、种子和输出目录再次训练会更新该组结果。完整流程使用 results/ 下的 compact、lightweight、serial 子目录保存 F、G、I，各目录使用同一划分。训练默认批量 64、AdamW、学习率 0.0003、预热 5 轮、余弦退火。

## 代码索引

| 文件或目录 | 内容 |
|---|---|
| models.py | 九组网络、Transformer 编码层、串行卷积制块器 |
| local_features.py | LSEVGG 的三阶段卷积、深度可分离卷积和 SE |
| data.py | 数据校验、固定分层划分、训练集归一化和增强 |
| experiment.py | prepare、train、analyze、all 入口；逐图预测、逐类指标、混淆矩阵 |
| analysis.py | 易混类别筛选、Attention Rollout、输入遮挡和错误案例图 |
| cnn_occlusion.py | 纯卷积 B/G 的 Grad-CAM 遮挡对照 |
| evaluate_compact.py | F/I 压缩实验、随机遮挡和 GPU 单张延迟；--serial 同时比较 F/I |
| summarize_results.py | 根据基础实验的逐组输出生成中文结果摘要 |
| reproduce.py | 顺序执行已完成组的训练及分析 |
| reports/seed42/ | 八组逐轮训练记录、参数、逐类指标、混淆矩阵（原始计数和行归一化） |
| reports/ | 详细 PDF、易混类别和遮挡统计、不同批次的延迟对照 |

区域遮挡使用工业区、住宅区及由 A 验证集选出的植被类别对。每次遮住 8/64 个图像块，随机对照重复 20 次。A/C/F/I 用 Attention Rollout，B/G 用 Grad-CAM。所有模型的随机遮挡沿用相同种子与样本顺序。比较延迟时优先使用同一轮测量的记录。
