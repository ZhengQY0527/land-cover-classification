"""Fit K's local residual readout to clean and masked training views with frozen C."""
import hashlib
import json
import random

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from data import Images
from fusion_trial import ROOT, RESULTS, masked
from models import Classifier


def main():
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    out = RESULTS / "readout_42" / "K_42"
    out.mkdir(parents=True, exist_ok=True)
    raw = (RESULTS / "manifest.json").read_bytes()
    ck = torch.load(RESULTS / "C_42/best.pt", map_location="cpu", weights_only=True)
    assert ck["manifest"] == hashlib.sha256(raw).hexdigest()
    model = Classifier("K").cuda().eval()
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    assert all(k.startswith("local_readout.") for k in missing) and len(missing)==4 and not unexpected
    model.requires_grad_(False)
    head = model.local_readout.requires_grad_(True)
    sets = {}
    for split in ("train","val"):
        ds = Images(ROOT / "data", json.loads(raw), split)
        rows = []
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            for x, y, _ in DataLoader(ds, batch_size=64):
                x = x.cuda()
                rank = torch.rand(len(y), 64, generator=generator).cuda()
                base, local = model(torch.cat([x, masked(x, rank)]), return_readout_inputs=True)
                rows.append((base, local, y.cuda().repeat(2), torch.arange(2,device=x.device).repeat_interleave(len(y))))
        sets[split] = [torch.cat(col) for col in zip(*rows)]
        print("features ready", split, len(ds), flush=True)
    logits, local, y, _ = sets["train"]
    v_logits, v_local, v_y, mode = sets["val"]
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    history = []
    best = -1
    for epoch in range(21):
        if epoch:
            for ix in torch.randperm(len(y), device=y.device).split(256):
                loss = torch.nn.functional.cross_entropy(logits[ix]+head(local[ix]),y[ix])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            p = (v_logits + head(v_local)).argmax(1)
            scores = [f1_score(v_y[mode==i].cpu(),p[mode==i].cpu(),average="macro") for i in (0,1)]
        score = sum(scores)/2
        history.append([epoch,*scores,score])
        if score > best:
            best = score
            tmp = out / "best.tmp"
            torch.save(dict(state=model.state_dict(), manifest=ck["manifest"],epoch=epoch),tmp)
            tmp.replace(out/"best.pt")
    pd.DataFrame(history,columns=["epoch","clean_macro_f1","random_macro_f1","mean"]).to_csv(out/"history.csv", index=False)
    print("selected", max(history,key=lambda r:r[-1]),flush=True)


if __name__ == "__main__":
    main()
