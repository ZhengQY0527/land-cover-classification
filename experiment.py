import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data import CLASSES, Images, prepare
from models import Classifier
from analysis import compare, interventions, plot_matrix


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    rows = []
    for x, y, paths in loader:
        logits = model(x.to(device)).cpu()
        probabilities = logits.softmax(1)
        for path, label, prob in zip(paths, y.tolist(), probabilities):
            rows.append((path, CLASSES[label], CLASSES[int(prob.argmax())], float(prob.max())))
    return pd.DataFrame(rows, columns=['path', 'truth', 'prediction', 'confidence'])


def score(frame):
    return f1_score(frame.truth, frame.prediction, labels=CLASSES, average='macro', zero_division=0)


def train(args, group, seed, folder, info, fingerprint, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loaders = {s: DataLoader(Images(args.data, info, s), args.batch, shuffle=s=='train',
                             num_workers=args.workers) for s in ('train', 'val', 'test')}
    model = Classifier(group).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.05)
    history = []
    best = -1
    checkpoint = folder/'best.pt'
    folder.mkdir(parents=True, exist_ok=True)
    warm = min(5, args.epochs)
    for epoch in range(args.epochs):
        factor = ((epoch+1)/warm if epoch < warm else
                  .5*(1+np.cos(np.pi*(epoch-warm+1)/(args.epochs-warm))))
        optimizer.param_groups[0]['lr'] = .0003 * max(float(factor), .001)
        model.train()
        total = 0.
        for x, y, _ in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()*len(y)
        val = predict(model, loaders['val'], device)
        metric = score(val)
        history.append(dict(epoch=epoch+1, loss=total/len(loaders['train'].dataset), val_f1=metric))
        print(group, seed, history[-1], flush=True)
        if metric > best:
            best = metric
            tmp = checkpoint.with_suffix('.tmp')
            torch.save(dict(state=model.state_dict(), manifest=fingerprint, group=group, seed=seed), tmp)
            tmp.replace(checkpoint)
            val.to_csv(folder/'validation.csv', index=False)
    pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['state'])
    frame = predict(model, loaders['test'], device)
    frame.to_csv(folder/'predictions.csv', index=False)
    cm = confusion_matrix(frame.truth, frame.prediction, labels=CLASSES)
    matrix = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    matrix.to_csv(folder/'confusion_counts.csv')
    normalized = matrix.div(matrix.sum(axis=1), axis=0)
    normalized.to_csv(folder/'confusion_rates.csv')
    plot_matrix(normalized, folder)
    pd.DataFrame(dict(category=CLASSES, support=cm.sum(1), recall=normalized.values.diagonal(),
                      f1=f1_score(frame.truth, frame.prediction, labels=CLASSES,
                                  average=None, zero_division=0))).to_csv(folder/'per_class.csv', index=False)
    return dict(group=group, seed=seed, accuracy=(frame.truth==frame.prediction).mean(),
                macro_f1=score(frame), parameters=sum(p.numel() for p in model.parameters()))


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'train', 'analyze', 'all'])
    p.add_argument('--data', type=Path, default=base/'data')
    p.add_argument('--out', type=Path, default=base/'results')
    p.add_argument('--groups', nargs='+', choices=list('ABCDEFGHI'), default=list('ABCDE'))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--random-masks', type=int, default=20)
    args = p.parse_args()
    if min(args.epochs, args.batch, args.random_masks) < 1:
        p.error('epochs, batch and random-masks must be positive')
    manifest = args.out/'manifest.json'
    if args.action in ('prepare', 'all'):
        prepare(args.data, manifest)
        if args.action == 'prepare':
            return
    info = json.loads(manifest.read_text(encoding='utf8'))
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.action in ('train', 'all'):
        rows = []
        for seed in args.seeds:
            for group in args.groups:
                folder = args.out/f'{group}_{seed}'
                result = train(args, group, seed, folder, info, fingerprint, device)
                result.update(epochs=args.epochs, batch=args.batch, manifest=fingerprint)
                (folder/'metrics.json').write_text(json.dumps(result), encoding='utf8')
                rows.append(result)
        df = pd.DataFrame(rows)
        df.to_csv(args.out/'runs.csv', index=False)
        df.groupby('group')[['accuracy','macro_f1']].agg(['mean','std']).to_csv(args.out/'summary.csv')
    if args.action in ('analyze', 'all'):
        pair = compare(args, info, fingerprint)
        for seed in args.seeds:
            interventions(args, info, fingerprint, seed, pair, device)


if __name__ == '__main__':
    main()
