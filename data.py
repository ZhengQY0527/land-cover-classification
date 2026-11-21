import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

CLASSES = sorted("AnnualCrop Forest HerbaceousVegetation Highway Industrial Pasture PermanentCrop Residential River SeaLake".split())


def read_image(path):
    with Image.open(path) as image:
        if image.size != (64, 64) or image.mode != 'RGB':
            raise ValueError(f"Expected 64x64 RGB: {path}")
        return image.copy()


def prepare(root, output):
    candidates = [p.parent for p in root.rglob('AnnualCrop') if p.is_dir()
                  and all((p.parent / c).is_dir() for c in CLASSES)]
    if len(candidates) != 1:
        raise ValueError(f"Expected one extracted RGB dataset under {root}, found {candidates}")
    splits = {s: [] for s in ('train', 'val', 'test')}
    rng = random.Random(42)
    for name in CLASSES:
        paths = sorted(p.relative_to(root).as_posix() for p in (candidates[0]/name).glob('*.jpg'))
        if len(paths) < 10:
            raise ValueError(f"Too few images in {name}")
        rng.shuffle(paths)
        a, b = round(.70*len(paths)), round(.85*len(paths))
        for split, subset in zip(splits, (paths[:a], paths[a:b], paths[b:])):
            splits[split].extend(subset)
    total = np.zeros(3)
    squares = np.zeros(3)
    for path in splits['train']:
        pixels = np.asarray(read_image(root/path), dtype=np.float64)/255
        total += pixels.mean((0, 1))
        squares += (pixels**2).mean((0, 1))
    n = len(splits['train'])
    mean = total/n
    info = dict(classes=CLASSES, splits=splits, mean=mean.tolist(),
                std=np.sqrt(np.maximum(squares/n-mean**2, 1e-12)).tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(info, indent=2), encoding='utf8')
    tmp.replace(output)


class Images(Dataset):
    def __init__(self, root, info, split):
        self.root, self.info, self.split = root, info, split
        self.paths = info['splits'][split]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        x = F.to_tensor(read_image(self.root/path))
        if self.split == 'train':
            if random.random() < .5:
                x = x.flip([2])
            x = x.rot90(random.randrange(4), [1, 2])
        return F.normalize(x, self.info['mean'], self.info['std']), CLASSES.index(Path(path).parent.name), path
