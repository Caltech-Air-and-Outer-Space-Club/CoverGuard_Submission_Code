"""Model factory for the cloud-gate ablation. Does not modify onboard_cloudscout.py (CloudGate keeps working).

variants: baseline (global max pool, 2-class CE) | A (mean+max pool) | B (max pool, fraction regression)
          | AB (dual pool, regression) | C (small U-Net, per-pixel mask; tile fraction = mean mask)
Every model returns raw output; `score(net, variant, out)` maps it to a per-tile cloudy score in [0,1].
"""
import torch, torch.nn as nn, torch.nn.functional as Fn


def cbr(i, o, k):
    return [nn.Conv2d(i, o, k, 1, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]


class DualPool(nn.Module):
    def __init__(self, mode="dual"):
        super().__init__(); self.mode = mode
    def forward(self, x):
        mean, mx = x.mean((2, 3)), x.amax((2, 3))
        return {"mean": mean, "max": mx, "dual": torch.cat([mean, mx], 1)}[self.mode]


class Classifier(nn.Module):
    """CloudScoutNet-style encoder; pool in {max, dual, mean}; n_out 2 (logits) or 1 (fraction logit)."""
    def __init__(self, in_ch=5, w=(32, 64, 96, 128), pool="max", n_out=2):
        super().__init__()
        L, c = cbr(in_ch, w[0], 5) + [nn.MaxPool2d(2)], w[0]
        for o in w:
            L += cbr(c, o, 3) + cbr(o, o, 1) + cbr(o, o, 3) + [nn.MaxPool2d(2)]
            c = o
        self.features = nn.Sequential(*L)
        self.pool = nn.AdaptiveMaxPool2d(1) if pool == "max" else DualPool(pool)
        self.flat = nn.Flatten() if pool == "max" else nn.Identity()
        k = 2 if pool == "dual" else 1
        self.decision = nn.Sequential(nn.Linear(c * k, 64), nn.ReLU(True), nn.Linear(64, n_out))
    def forward(self, x):
        return self.decision(self.flat(self.pool(self.features(x))))


class UNet(nn.Module):
    """Small U-Net: 4 encoder stages, bilinear upsampling with skips, 1-channel mask logit at input resolution."""
    def __init__(self, in_ch=5, w=(16, 32, 64, 96)):
        super().__init__()
        self.enc = nn.ModuleList(); c = in_ch
        for o in w:
            self.enc.append(nn.Sequential(*cbr(c, o, 3), *cbr(o, o, 3))); c = o
        self.dec = nn.ModuleList()
        for o in reversed(w[:-1]):
            self.dec.append(nn.Sequential(*cbr(c + o, o, 3), *cbr(o, o, 3))); c = o
        self.out = nn.Conv2d(c, 1, 1)
    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x); x = Fn.max_pool2d(x, 2)
        for d in self.dec:
            x = Fn.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = d(torch.cat([x, skips.pop()], 1))
        return self.out(x)


def build(variant):
    return {"baseline": lambda: Classifier(pool="max", n_out=2), "A": lambda: Classifier(pool="dual", n_out=2),
            "B": lambda: Classifier(pool="max", n_out=1), "AB": lambda: Classifier(pool="dual", n_out=1),
            "C": lambda: UNet()}[variant]()


def score(variant, out):
    """per-tile cloudy score in [0,1] from the raw model output"""
    if variant in ("baseline", "A"):
        return torch.softmax(out, 1)[:, 1]
    if variant in ("B", "AB"):
        return torch.sigmoid(out[:, 0])
    return torch.sigmoid(out[:, 0]).mean((1, 2))        # C: tile fraction = mean of the mask probabilities
