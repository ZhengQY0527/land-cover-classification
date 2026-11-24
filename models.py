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



class SpatialEncoder(Encoder):
    """Mix adjacent patch features in the expanded feed-forward representation."""
    def __init__(self, dim):
        super().__init__(dim)
        hidden = dim * 2
        self.neighborhood = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        nn.init.zeros_(self.neighborhood.weight)
        nn.init.zeros_(self.neighborhood.bias)

    def forward(self, x, explain):
        norm = self.norm1(x)
        delta, weights = self.attention(norm, norm, norm, need_weights=explain)
        x = x + delta
        hidden = self.mlp[1](self.mlp[0](self.norm2(x)))
        patches = hidden[:, 1:].transpose(1, 2).reshape(len(x), -1, 8, 8)
        patches = patches + self.neighborhood(patches)
        mixed = torch.cat((hidden[:, :1], patches.flatten(2).transpose(1, 2)), dim=1)
        return x + self.mlp[2](mixed), weights


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


class PatchReadout(nn.Module):
    def __init__(self, dim, attention):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False) if attention else None
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, 1)
        self.predict = nn.Linear(dim, 10)
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)
        if attention:
            nn.init.zeros_(self.score.weight)
        self.register_buffer("scale", torch.tensor(1.))

    def forward(self, patches):
        patches = self.norm(patches)
        if self.score is None:
            pooled = patches.mean(1)
        else:
            pooled = (self.score(patches).softmax(1) * patches).sum(1)
        return self.scale * self.gate(pooled).sigmoid() * self.predict(pooled)


class Classifier(nn.Module):
    def __init__(self, group):
        super().__init__()
        if group not in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'):
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
            encoder = SpatialEncoder if group == 'J' else Encoder
            self.layers = nn.ModuleList(encoder(dim) for _ in range(depth))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 10))
        if group in ('M','N'):
            self.patch_readout = PatchReadout(dim, attention=group=='N')
        if group == 'L':
            self.register_buffer('readout_threshold', torch.tensor(1.))
            self.register_buffer('readout_scale', torch.tensor(1.))
        if group in ('K','L'):
            self.local_readout = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 10))
            nn.init.zeros_(self.local_readout[1].weight)
            nn.init.zeros_(self.local_readout[1].bias)

    def forward(self, x, explain=False, return_readout_inputs=False):
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
        if self.group in ('M','N'):
            if return_readout_inputs:
                return logits, tokens[:, 1:]
            logits = logits + self.patch_readout(tokens[:, 1:])
        if self.group in ('K','L'):
            pooled = local.mean((2, 3))
            if return_readout_inputs:
                return logits, pooled
            correction = self.local_readout(pooled)
            if self.group == 'L':
                confidence = logits.softmax(1).amax(1, keepdim=True)
                correction = correction * (confidence <= self.readout_threshold) * self.readout_scale
            logits = logits + correction
        return (logits, rollout[:, 0, 1:]) if explain else logits
