import os
import json
from typing import List
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from vlm.config import CFG


class SimpleTokenizer:
    """
    Very simple whitespace tokenizer with frequency-based vocab.
    """
    def __init__(self, texts: List[str], min_freq: int = 1):
        counter = Counter()
        for t in texts:
            counter.update(t.lower().strip().split())

        vocab = ["<pad>", "<unk>"]
        for w, c in counter.items():
            if c >= min_freq:
                vocab.append(w)

        self.itos = vocab
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.pad_idx = self.stoi["<pad>"]
        self.unk_idx = self.stoi["<unk>"]
        self.vocab_size = len(self.itos)

    def encode(self, text: str, max_len: int):
        tokens = text.lower().strip().split()
        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokens]
        if len(ids) < max_len:
            ids = ids + [self.pad_idx] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids


class PusherVLMDataset(Dataset):
    """
    Returns (image_tensor, token_ids, goal_xy).
    CLIP will use (image, token_ids); goal_xy can be used later.
    """
    def __init__(self, root_dir, tokenizer: SimpleTokenizer, max_len, img_size):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(os.path.join(root_dir, "metadata.json"), "r") as f:
            self.meta = json.load(f)

        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]

        img_path = os.path.join(self.root_dir, item["image"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        token_ids = torch.tensor(
            self.tokenizer.encode(item["text"], self.max_len),
            dtype=torch.long,
        )

        goal = torch.tensor(item["goal"], dtype=torch.float32)  # (x, y)

        return img, token_ids, goal


def build_tokenizer_and_dataset(cfg=CFG):
    meta_path = os.path.join(cfg.dataset_dir, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    all_texts = [m["text"] for m in meta]
    tokenizer = SimpleTokenizer(all_texts, min_freq=1)

    dataset = PusherVLMDataset(
        root_dir=cfg.dataset_dir,
        tokenizer=tokenizer,
        max_len=cfg.max_text_len,
        img_size=cfg.img_size,
    )

    return tokenizer, dataset
