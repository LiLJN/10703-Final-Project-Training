import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm.config import CFG
from vlm.dataset import build_tokenizer_and_dataset
from vlm.model import ClipDualEncoder


def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    """
    Standard CLIP InfoNCE loss over a batch of N (image, text) pairs.
    """
    batch_size = image_embeds.size(0)

    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()

    labels = torch.arange(batch_size, device=image_embeds.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2.0
    return loss


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def train_clip_two_policies(cfg=CFG):
    """
    Train CLIP dual encoder with two separate "policies":

      - Vision policy: updates only the vision encoder parameters
      - Language policy: updates only the text encoder parameters

    Both are trained with the same CLIP contrastive objective, but in
    two distinct steps per batch.
    """
    tokenizer, dataset = build_tokenizer_and_dataset(cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    device = torch.device(cfg.device)
    model = ClipDualEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        text_hidden_dim=256,
    ).to(device)

    # Separate optimizers (policies)
    vision_params = list(model.image_encoder.parameters()) + [model.logit_scale]
    text_params = list(model.text_encoder.parameters())

    opt_vision = torch.optim.AdamW(vision_params, lr=cfg.lr_vision, weight_decay=1e-4)
    opt_text = torch.optim.AdamW(text_params, lr=cfg.lr_text, weight_decay=1e-4)

    print(f"[train_clip_two_policies] Training on {len(dataset)} samples, device={device}")

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss_vision = 0.0
        total_loss_text = 0.0
        total_count = 0

        for images, token_ids, goals in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            token_ids = token_ids.to(device)

            batch_size = images.size(0)
            total_count += batch_size

            # ---- Step 1: Vision policy update (freeze text encoder) ----
            set_requires_grad(model.text_encoder, False)
            set_requires_grad(model.image_encoder, True)

            opt_vision.zero_grad()

            img_emb, txt_emb, logit_scale = model(images, token_ids)
            loss_v = clip_contrastive_loss(img_emb, txt_emb.detach(), logit_scale)
            loss_v.backward()
            opt_vision.step()

            total_loss_vision += loss_v.item() * batch_size

            # ---- Step 2: Text policy update (freeze vision encoder) ----
            set_requires_grad(model.text_encoder, True)
            set_requires_grad(model.image_encoder, False)

            opt_text.zero_grad()

            img_emb, txt_emb, logit_scale = model(images, token_ids)
            loss_t = clip_contrastive_loss(img_emb.detach(), txt_emb, logit_scale)
            loss_t.backward()
            opt_text.step()

            total_loss_text += loss_t.item() * batch_size

            # Re-enable grads for next iteration safety
            set_requires_grad(model.image_encoder, True)
            set_requires_grad(model.text_encoder, True)

        avg_loss_v = total_loss_vision / total_count
        avg_loss_t = total_loss_text / total_count
        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} - "
            f"VisionLoss: {avg_loss_v:.6f} | TextLoss: {avg_loss_t:.6f} | "
            f"logit_scale: {model.logit_scale.exp().item():.3f}"
        )

    ckpt_path = os.path.join(cfg.dataset_dir, "pusher_clip_two_policies.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": tokenizer.itos,
            "config": vars(cfg),
        },
        ckpt_path,
    )
    print(f"[train_clip_two_policies] Model saved to {ckpt_path}")

    return model, tokenizer


def evaluate_retrieval(model, tokenizer, cfg=CFG, num_samples: int = 32):
    """
    Simple image-text retrieval sanity check:
      - Take N examples.
      - Compute similarity matrix.
      - Report how often the correct text is top-1 for each image.
    """
    from vlm.dataset import PusherVLMDataset

    import json

    device = torch.device(cfg.device)

    meta_path = os.path.join(cfg.dataset_dir, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    meta_small = meta[:num_samples]

    transform = torch.nn.Sequential()  # not used, we reuse dataset transform

    from vlm.dataset import PusherVLMDataset
    small_dataset = PusherVLMDataset(
        root_dir=cfg.dataset_dir,
        tokenizer=tokenizer,
        max_len=cfg.max_text_len,
        img_size=cfg.img_size,
    )

    # Just take first num_samples
    imgs = []
    txts = []
    for i in range(num_samples):
        img, token_ids, goal = small_dataset[i]
        imgs.append(img.unsqueeze(0))
        txts.append(token_ids.unsqueeze(0))

    imgs = torch.cat(imgs, dim=0).to(device)       # (N, 3, H, W)
    txts = torch.cat(txts, dim=0).to(device)       # (N, T)

    model.eval()
    with torch.no_grad():
        img_emb, txt_emb, logit_scale = model(imgs, txts)
        sims = logit_scale * img_emb @ txt_emb.t()  # (N, N)

    # For each image i, find the index of the best-matching text
    top1 = sims.argmax(dim=1)   # (N,)
    correct = (top1 == torch.arange(num_samples, device=device)).sum().item()
    acc = correct / num_samples

    print(f"[evaluate_retrieval] Top-1 image→text accuracy on {num_samples} samples: {acc*100:.2f}%")
    print("Some matches (image_idx: best_text_idx):")
    for i in range(min(5, num_samples)):
        print(f"  {i}: {top1[i].item()}")