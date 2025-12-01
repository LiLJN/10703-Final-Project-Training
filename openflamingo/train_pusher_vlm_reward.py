# train_pusher_vlm_reward.py
import os
import json
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

PROMPT = "<image>In this Pusher state, the immediate reward is "

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class PusherRewardDataset(Dataset):
    def __init__(self, root_dir, image_processor, tokenizer, max_length=32):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        meta_path = os.path.join(root_dir, "meta.jsonl")
        with open(meta_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_path = os.path.join(self.image_dir, rec["image"])
        reward = rec["reward"]

        image = Image.open(img_path).convert("RGB")
        img_tensor = self.image_processor(image)  # (3, H, W)

        reward_text = f"{reward:.2f}"
        full_text = PROMPT + reward_text + self.tokenizer.eos_token

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        prompt_enc = self.tokenizer(PROMPT, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100 

        return {
            "image": img_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def collate_fn(batch, tokenizer):
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    vision_x = imgs.unsqueeze(1).unsqueeze(2)
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,
        return_tensors="pt",
    )
    padded_labels = tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    return {
        "vision_x": vision_x,
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "labels": padded_labels,
    }

def load_openflamingo(device):
    print("Loading OpenFlamingo...")
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
    )
    ckpt_path = hf_hub_download(
        "openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt"
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.train()
    if hasattr(model, "vision_encoder"):
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        print("Froze vision encoder parameters.")
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Total params: {total_params:,}, "
        f"Trainable params: {trainable_params:,}"
    )
    return model, image_processor, tokenizer

def train(
    data_dir="pusher_vlm_data",
    output_dir="pusher_vlm_ckpts",
    batch_size=8,
    num_epochs=3,
    lr=1e-5,
    ):
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    model, image_processor, tokenizer = load_openflamingo(device)

    dataset = PusherRewardDataset(
        data_dir, image_processor, tokenizer, max_length=32
    )
    collate = partial(collate_fn, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    global_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                vision_x=batch["vision_x"],
                lang_x=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                avg = running_loss / 50
                print(f"Step {global_step}, avg loss: {avg:.4f}")
                running_loss = 0.0

        ckpt_path = os.path.join(output_dir, f"vlm_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    final_path = os.path.join(output_dir, "vlm_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    train(
        data_dir="pusher_vlm_data",
        output_dir="pusher_vlm_ckpts",
        batch_size=4,   
        num_epochs=2,   
        lr=1e-5,
    )
