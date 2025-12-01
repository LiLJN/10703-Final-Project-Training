import torch
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

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