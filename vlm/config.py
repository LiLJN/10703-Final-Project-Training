from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class Config:
    # Gym environment
    env_id: str = "Pusher-v5"

    # Dataset
    dataset_dir: str = "pusher_vlm_dataset_v5_clip"
    num_samples: int = 2000

    # Rendering / images
    img_size: Tuple[int, int] = (224, 224)     # model input
    camera_size: Tuple[int, int] = (256, 256)  # raw render size

    # Training
    batch_size: int = 64
    num_epochs: int = 10
    lr_vision: float = 1e-4
    lr_text: float = 1e-4
    max_text_len: int = 16
    embed_dim: int = 256

    # Camera pose for top-down view
    camera_distance: float = 3.0
    camera_elevation: float = -90.0
    camera_azimuth: float = 90.0
    camera_lookat: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
