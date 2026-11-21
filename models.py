import torch
from torch import nn
from local_features import LocalFeatures, DepthwiseSeparableConv, SEBlock


class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x, explain):
        norm = self.norm1(x)
        delta, weights = self.attention(norm, norm, norm, need_weights=explain)
        x = x + delta
        return x + self.mlp(self.norm2(x)), weights



class ConvTokenizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(),
                                  DepthwiseSeparableConv(32, 64, stride=2))
        self.texture = nn.Sequential(DepthwiseSeparableConv(64, 64), SEBlock(64))
        self.tokens = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(64, dim, 1))

    def forward(self, x):
        x = self.stem(x)
        return self.tokens(x + self.texture(x))


class Classifier(nn.Module):
    def __init__(self, group):
        super().__init__()
        if group not in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'):
            raise ValueError(group)
        self.group = group
        dim = 96 if group in ("H", "I") else 128
        if group not in ('A', 'I'):
            width = 192 if group in ('G', 'H') else 256
            self.local = LocalFeatures(width)
            self.project = nn.Conv2d(width, dim, 1)
        if group not in ('B', 'G'):
            self.patch = ConvTokenizer(dim) if group == "I" else nn.Conv2d(3, dim, 8, stride=8)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            if group != 'D':
                self.position = nn.Parameter(torch.empty(1, dim, 8, 8))
                nn.init.trunc_normal_(self.position, std=.02)
            depth = 2 if group in ('F', 'H') else 4
            if group == "I":
                depth = 3
            self.layers = nn.ModuleList(Encoder(dim) for _ in range(depth))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 10))

    def forward(self, x, explain=False):
        if self.group not in ('A', 'I'):
            local = self.project(self.local(x))
            if self.group in ('B', 'G'):
                return self.head(local.mean((2, 3)))
        grid = self.patch(x)
        if self.group not in ('A', 'I'):
            grid = grid + (local.mean((2, 3), keepdim=True) if self.group == 'E' else local)
        if self.group != 'D':
            grid = grid + self.position
        tokens = torch.cat((self.cls.expand(len(x), -1, -1), grid.flatten(2).transpose(1, 2)), 1)
        rollout = None
        for layer in self.layers:
            tokens, weights = layer(tokens, explain)
            if explain:
                weights = weights + torch.eye(tokens.size(1), device=x.device)
                weights = weights / weights.sum(-1, keepdim=True)
                rollout = weights if rollout is None else weights @ rollout
        logits = self.head(tokens[:, 0])
        return (logits, rollout[:, 0, 1:]) if explain else logits
