import gymnasium as gym
from tqdm import tqdm
from PIL import Image
import os
import json
import argparse
import random
import numpy as np

def generate_random_goalpoint():
    x = 10
    y = 10
    while x**2 + y**2 > 1.0:
        x = random.uniform(-1, 1)
        y = random.uniform(-0.5, 0.5)
    return np.asarray([x,y])

def collect_pusher_dataset(
    save_dir: str = "pusher_vlm_data",
    num_episodes: int = 2000,
    max_steps: int = 200,
):
    env = gym.make("Pusher-v5", render_mode="rgb_array")

    
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    meta_path = os.path.join(save_dir, "meta.jsonl")

    total_samples = 0

    with open(meta_path, "w") as f:
        for ep in tqdm(range(num_episodes), desc="Collecting Episodes"):
            obs, info = env.reset()
            done = False

            model = env.unwrapped.model
            new_goal_xy = generate_random_goalpoint()
            goal_body_id = None
            for i in range(model.nbody):
                name = model.body(i).name
                if name is not None and "goal" in name.lower():
                    goal_body_id = i
            model.body_pos[goal_body_id, 0] = new_goal_xy[0]
            model.body_pos[goal_body_id, 1] = new_goal_xy[1]
            
            for step in tqdm(range(max_steps), desc=f"Ep {ep}", leave=False):
                if done:
                    break

                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                frame = env.render() 
                img = Image.fromarray(frame)

                img_name = f"ep{ep:04d}_step{step:03d}.png"
                img_path = os.path.join(img_dir, img_name)
                img.save(img_path)

                record = {
                    "image": img_name,
                    "reward": float(reward),
                    "episode": ep,
                    "step": step,
                    
                    "goal_xy": new_goal_xy.tolist(),
                }
                f.write(json.dumps(record) + "\n")

                total_samples += 1

    env.close()
    print(f"\nDataset collection complete! Saved {total_samples} samples to {save_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="pusher_vlm_data")
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()

    collect_pusher_dataset(
        save_dir=args.save_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )