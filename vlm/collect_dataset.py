import os
import json
import random  # <-- use Python's random instead of numpy for text choice

from tqdm import tqdm
from PIL import Image
import numpy as np
import gymnasium as gym
import mujoco

from vlm.config import CFG


PROMPTS = [
    "Place the object at the goal position.",
    "Push the object so that it reaches the goal.",
    "Move the object until it lands on the target.",
    "Guide the object to the goal location.",
]

def render_top_down(env, width: int, height: int, cfg=CFG):
    model = env.unwrapped.model
    data = env.unwrapped.data

    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)

    cam.lookat[:] = np.array(cfg.camera_lookat, dtype=float)
    cam.distance = float(cfg.camera_distance)
    cam.azimuth = float(cfg.camera_azimuth)
    cam.elevation = float(cfg.camera_elevation)

    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    renderer.close()
    return img


def collect_dataset(cfg=CFG):
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    img_dir = os.path.join(cfg.dataset_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    metadata_path = os.path.join(cfg.dataset_dir, "metadata.json")
    if os.path.exists(metadata_path):
        print(f"[collect_dataset] {metadata_path} already exists, skipping.")
        return

    env = gym.make(cfg.env_id)
    metadata = []

    print(f"[collect_dataset] Generating {cfg.num_samples} samples into {cfg.dataset_dir}...")
    for i in tqdm(range(cfg.num_samples), desc="Collecting Pusher CLIP dataset"):
        obs, info = env.reset()

        goal_xyz = obs[20:23].astype(float).tolist()
        goal_xy = goal_xyz[:2]  # [x, y]
        

        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

        frame = render_top_down(env, cfg.camera_size[0], cfg.camera_size[1], cfg)
        img = Image.fromarray(frame)

        img_name = f"pusher_{i:06d}.png"
        rel_img_path = os.path.join("images", img_name)
        abs_img_path = os.path.join(cfg.dataset_dir, rel_img_path)
        img.save(abs_img_path)

        text_prompt = random.choice(PROMPTS)
        text_prompt = str(text_prompt)  # just to be explicit

        metadata.append(
            {
                "image": rel_img_path,
                "text": text_prompt,
                "goal": [float(goal_xy[0]), float(goal_xy[1])],
            }
        )

    env.close()

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[collect_dataset] Saved {len(metadata)} samples to {cfg.dataset_dir}")
