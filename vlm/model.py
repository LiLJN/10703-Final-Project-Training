import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, token_ids):
        x = self.embed(token_ids)        # (B, T, E)
        _, h_n = self.gru(x)            # (1, B, H)
        return h_n.squeeze(0)           # (B, H)


class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = backbone.fc.in_features
        self.proj = nn.Linear(self.out_dim, embed_dim)

    def forward(self, images):
        feat = self.backbone(images)           # (B, C, 1, 1)
        feat = feat.view(feat.size(0), -1)     # (B, C)
        feat = self.proj(feat)                 # (B, D)
        feat = F.normalize(feat, dim=-1)
        return feat


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, text_hidden_dim=256):
        super().__init__()
        self.text_encoder = SimpleTextEncoder(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=text_hidden_dim,
        )
        self.proj = nn.Linear(text_hidden_dim, embed_dim)

    def forward(self, token_ids):
        feat = self.text_encoder(token_ids)    # (B, H)
        feat = self.proj(feat)                 # (B, D)
        feat = F.normalize(feat, dim=-1)
        return feat


class ClipDualEncoder(nn.Module):
    """
    CLIP-style dual encoder for Pusher:

        image -> vision encoder -> embedding
        text  -> text encoder   -> embedding

    We also keep logit_scale as in CLIP.
    """
    def __init__(self, vocab_size, embed_dim=256, text_hidden_dim=256):
        super().__init__()

        self.image_encoder = VisionEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_hidden_dim=text_hidden_dim,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images, token_ids):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(token_ids)
        logit_scale = self.logit_scale.exp()
        return img_emb, txt_emb, logit_scale
