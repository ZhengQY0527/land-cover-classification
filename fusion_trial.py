"""Seed-42 decision fusion. Run with python fusion_trial.py from any directory."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import Images, CLASSES
from models import Classifier

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "fusion_42"
NAMES = ["B", "C", "J", "equal", "gate"]


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 3)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, p):
        top = p.topk(2).values
        features = torch.cat([top[:, :, 0], top[:, :, 0] - top[:, :, 1]], dim=1)
        weights = self.linear(features).softmax(1)
        return (weights.unsqueeze(-1) * p).sum(1)


def masked(x, rank, high=True):
    indices = rank.topk(8, largest=high).indices
    mask = torch.zeros_like(rank).scatter_(1, indices, 1).reshape(-1, 1, 8, 8)
    return x * (1 - mask.repeat_interleave(8, 2).repeat_interleave(8, 3))


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    raw = (RESULTS / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    info = json.loads(raw)
    experts = []
    for name, folder in zip(NAMES, ["B_42", "C_42", "spatial_resumed/J_42"]):
        ck = torch.load(RESULTS / folder / "best.pt", map_location="cpu", weights_only=True)
        assert ck["manifest"] == digest
        model = Classifier(name).cuda().eval()
        model.load_state_dict(ck["state"])
        experts.append(model)

    @torch.no_grad()
    def probabilities(x):
        return torch.stack([m(x).softmax(1) for m in experts], 1)

    val = Images(ROOT / "data", info, "val")
    generator = torch.Generator().manual_seed(42)
    clean, occluded, labels = [], [], []
    for x, y, _ in DataLoader(val, batch_size=64):
        x = x.cuda()
        clean.append(probabilities(x).cpu())
        occluded.append(probabilities(masked(x, torch.rand(len(y), 64, generator=generator).cuda())).cpu())
        labels.append(y)
    clean, occluded, labels = map(torch.cat, (clean, occluded, labels))
    fit, hold = train_test_split(np.arange(len(val)), test_size=0.5, random_state=42, stratify=labels)
    gate = Gate()
    opt = torch.optim.Adam(gate.parameters(), lr=.03)
    train_p = torch.cat([clean[fit], occluded[fit]])
    truth = labels[fit].repeat(2)
    for _ in range(200):
        opt.zero_grad()
        loss = nn.functional.nll_loss(gate(train_p).clamp_min(1e-10).log(), truth)
        (loss + .0002 * gate.linear.weight.square().mean()).backward()
        opt.step()
    gate.eval()

    def outputs(p):
        return [*p.unbind(1), p.mean(1), gate(p)]

    with torch.no_grad():
        scores = {}
        for mode, p in [("clean", clean), ("random", occluded)]:
            scores[mode] = [f1_score(labels[hold], s.argmax(1), average="macro") for s in outputs(p[hold])]
        frame = pd.DataFrame(scores, index=NAMES)
        frame["mean"] = frame.mean(axis=1)
        chosen = frame.loc[["equal", "gate"], "mean"].idxmax()
        frame.to_csv(OUT / "validation.csv")
        torch.save(dict(state=gate.state_dict(), selected=chosen, manifest=digest, experts=NAMES[:3]), OUT / "gate.pt")
        print(frame, "selected", chosen, flush=True)
        gate.cuda()
        test = Images(ROOT / "data", info, "test")
        pred_rows = []
        for x, y, paths in DataLoader(test, batch_size=64):
            x = x.cuda()
            p = probabilities(x)
            for name, s in zip(NAMES, outputs(p)):
                pred_rows.extend(zip(paths, y.tolist(), [name]*len(y), s.argmax(1).tolist()))
        pred = pd.DataFrame(pred_rows, columns=["path","truth","model","prediction"])
        pred.to_csv(OUT / "predictions.csv", index=False)
        for name, part in pred.groupby("model"):
            pd.DataFrame(confusion_matrix(part.truth, part.prediction, labels=range(10)), index=CLASSES, columns=CLASSES).to_csv(OUT / (name + "_confusion.csv"))
        focus = set(pd.read_csv(RESULTS / "occlusion_42.csv").path)
        subset = Subset(test, [i for i, path in enumerate(test.paths) if path in focus])
        assert len(subset) == len(focus) == 1650
        generator.manual_seed(42)
        counts = {mode: np.zeros(len(NAMES), dtype=int) for mode in ["clean","high","low","random"]}
        for x, y, _ in DataLoader(subset, batch_size=32):
            x, y = x.cuda(), y.cuda()
            _, attention = experts[1](x, explain=True)
            for mode in counts:
                for _ in range(20 if mode == "random" else 1):
                    view = x
                    if mode == "random":
                        view = masked(x, torch.rand(len(y), 64, generator=generator).cuda())
                    elif mode != "clean":
                        view = masked(x, attention, mode == "high")
                    counts[mode] += torch.stack([(s.argmax(1)==y).sum() for s in outputs(probabilities(view))]).cpu().numpy()
        rows = []
        for i, name in enumerate(NAMES):
            part = pred[pred.model == name]
            row = dict(model=name, correct=int((part.truth==part.prediction).sum()), macro_f1=f1_score(part.truth,part.prediction,average="macro"))
            row.update({mode:int(hits[i]) for mode,hits in counts.items()})
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(OUT / "test.csv", index=False)
        print(summary, flush=True)


if __name__ == "__main__":
    main()
