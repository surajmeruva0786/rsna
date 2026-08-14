"""Study-level multi-label model for knee MRI.

A knee study is a *bag of series*, not a single image: each of the six plane x
fluid-sensitivity slots shows a different subset of the twelve findings (sagittal
fluid-sensitive carries the menisci and ACL, axial carries the patellofemoral joint and
Baker's cyst, and so on). The model therefore encodes each slot independently and pools
across whichever slots the study actually has.

Each slot is fed to the backbone as a *16-channel image* rather than as 16 separate
forward passes. This is the decisive choice for a 4 GB Pascal card: it cuts the number
of backbone evaluations per study from ~96 to <=6 while still letting the first
convolution mix across the slice axis, which is where through-plane continuity of a
tear actually lives.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SLOTS = 6


class SlotAttentionPool(nn.Module):
    """Masked attention pooling over the (<=6) present slots of a study.

    Mean pooling would dilute the one informative plane with up to five uninformative
    ones, and max pooling is unstable under the weak labels. Gated attention lets the
    network learn, per finding, which plane to trust -- and the mask keeps absent slots
    from contributing at all, rather than contributing a zero vector that would drag
    the mean toward zero for studies with fewer series.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)   mask: (B, S) with 1 for a present slot
        w = self.score(self.attn(x) * self.gate(x)).squeeze(-1)  # (B, S)
        w = w.masked_fill(mask < 0.5, float("-inf"))
        # A study with no usable series at all would otherwise softmax over all -inf
        # and produce NaN; fall back to uniform weights over the (empty) bag.
        empty = mask.sum(dim=1, keepdim=True) < 0.5
        w = torch.where(empty.expand_as(w), torch.zeros_like(w), w)
        return torch.einsum("bs,bsd->bd", F.softmax(w, dim=1), x)


class KneeNet(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        n_slices: int = 16,
        n_targets: int = 12,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=n_slices, num_classes=0
        )
        dim = self.backbone.num_features

        # Which plane/contrast a feature came from is information the pooling layer
        # needs; without it the attention head cannot distinguish a sagittal slot from
        # an axial one when deciding what to weight.
        self.slot_embed = nn.Parameter(torch.zeros(N_SLOTS, dim))
        nn.init.trunc_normal_(self.slot_embed, std=0.02)

        self.pool = SlotAttentionPool(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_targets))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C, H, W)   mask: (B, S)
        b, s = x.shape[:2]
        feat = self.backbone(x.flatten(0, 1)).view(b, s, -1)
        feat = feat + self.slot_embed.unsqueeze(0)
        return self.head(self.pool(feat, mask))
