"""Generate the Chinese analysis from completed EuroSAT runs."""
import json
from pathlib import Path
import pandas as pd


def report(out):
    runs = pd.read_csv(out/'runs.csv')
    pairs = pd.read_csv(out/'confused_pairs.csv')
    seeds = sorted(runs.seed.unique())
    occlusion = pd.concat([pd.read_csv(out/f'occlusion_summary_{s}.csv').assign(seed=s) for s in seeds])
    scores = runs.pivot(index='seed', columns='group', values='macro_f1')
    lines = ['# EuroSAT RGB 实验分析', '',
             '使用固定图像级划分；训练 18,900 张、验证 4,050 张、测试 4,050 张。每组按验证集宏 F1 选择检查点。以下为已完成种子（' + ', '.join(map(str, seeds)) + '）的测试集结果。',
             '', '|组别|平均准确率|宏 F1（多种子附样本标准差）|参数量|','|---|---:|---:|---:|']
    for group, block in runs.groupby('group'):
        spread = f' ± {block.macro_f1.std():.4f}' if len(block)>1 else ''
        lines.append(f'|{group}|{block.accuracy.mean():.2%}|{block.macro_f1.mean():.4f}{spread}|{block.parameters.iloc[0]:,}|')
    lines.extend(['','## 融合和消融',''])
    for other, label in [('A','引入卷积分支'),('B','融合模型与卷积对照'),('D','显式位置编码'),('E','逐位置对齐')]:
        delta = (scores.C-scores[other])*100
        lines.append(f'- {label}（C 减 {other}）：宏 F1 平均差值 {delta.mean():+.2f} 个百分点，{(delta>0).sum()}/{len(delta)} 个种子为正。')
    lines.extend(['','## 相似类别的双向误判','',
                  '|真实类 → 错分类|A 平均误判率|C 平均误判率|变化（百分点）|','|---|---:|---:|---:|'])
    for (src,dst),block in pairs.groupby(['truth','confused_with']):
        rates=block.groupby('group').rate.mean()
        lines.append(f'|{src} → {dst}|{rates.A:.2%}|{rates.C:.2%}|{100*(rates.C-rates.A):+.2f}|')
    lines.extend(['','## 区域干预','',
                  '每次遮挡 8/64 个块。下表显示真实类别 logit 的平均下降（高响应遮挡减随机遮挡）。正值表示遮挡高响应位置造成更大下降。不同模型的原始 logit 尺度可能不同，宜优先作模型内部比较。',''])
    for (group,cat),block in occlusion.groupby(['group','truth']):
        values=block.groupby('mask').logit_drop.mean()
        lines.append(f'- {group} / {cat}：{values.high-values.random:+.3f}，高响应减低响应为 {values.high-values.low:+.3f}。')
    lines.extend(['','典型图像见各 A/C 组目录中的 corrected、regressed 和 both_wrong 图片。图像级划分只衡量本次划分的泛化表现，跨地区迁移仍需独立验证。'])
    (out/'实验分析.md').write_text(chr(10).join(lines),encoding='utf8')


if __name__ == '__main__':
    report(Path(__file__).resolve().parent/'results')
