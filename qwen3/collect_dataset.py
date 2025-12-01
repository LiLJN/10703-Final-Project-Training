import gymnasium as gym
from tqdm import tqdm
from PIL import Image
import os
import json
import argparse
import random
import numpy as np
import mujoco
from stable_baselines3 import SAC

def generate_random_goalpoint():
    x = 10
    y = 10
    while x**2 + y**2 > 1.0:
        x = random.uniform(-1, 1)
        y = random.uniform(-0.5, 0.5)
    return np.asarray([x, y])

def balanced_binary_array(n):
    arr = np.array([0] * (n//2) + [1] * (n//2))
    np.random.shuffle(arr)
    return arr

def collect_pusher_dataset(
    save_dir: str = "pusher_vlm_data",
    num_episodes: int = 2000,
    max_steps: int = 200,
    model_path: str = "sac_pusher.zip",
):
    env = gym.make("Pusher-v5", render_mode="rgb_array")
    policy_model = SAC.load(model_path, env=env)
    
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    meta_path = os.path.join(save_dir, "meta.jsonl")

    total_samples = 0
    toggle_random_action = balanced_binary_array(num_episodes)

    with open(meta_path, "w") as f:
        for ep in tqdm(range(num_episodes), desc="Collecting Episodes"): 
            toggle_random_action_flag = toggle_random_action[ep]
            obs, _ = env.reset()
            done = False

            model = env.unwrapped.model
            new_goal_xy = generate_random_goalpoint()
            goal_body_id = None
            for i in range(model.nbody):
                name = model.body(i).name
                if name is not None and "goal" in name.lower():
                    goal_body_id = i
                    break
            if goal_body_id is None:
                raise RuntimeError("Could not find goal body in model.")
            model.body_pos[goal_body_id, 0] = new_goal_xy[0]
            model.body_pos[goal_body_id, 1] = new_goal_xy[1]

            mujoco.mj_forward(model, env.unwrapped.data)
            prev_obs = env.unwrapped._get_obs()
            frame = env.render()
            img_pil = Image.fromarray(frame)
            
            step_count = 0
            img_name = f"ep{ep:04d}_step{step_count:03d}.png"
            img_path = os.path.join(img_dir, img_name)
            img_pil.save(img_path)
            
            fingertip_obs = np.asarray(prev_obs, dtype=float).tolist()[14:17]
            object_obs = np.asarray(prev_obs, dtype=float).tolist()[17:20]
            goal_obs = np.asarray(prev_obs, dtype=float).tolist()[20:]
            
            for _ in tqdm(range(max_steps-1), desc=f"Ep {ep}", leave=False):
                if done:
                    break
                
                if toggle_random_action_flag == 0:
                    # print("Using random action")
                    action = env.action_space.sample()
                else:
                    # print("Using policy action")
                    action, _ = policy_model.predict(prev_obs, deterministic=True)

                curr_obs, reward, terminated, _, _ = env.step(action)
                step_count += 1
                done = terminated
                curr_fingertip_obs = np.asarray(curr_obs, dtype=float).tolist()[14:17]
                curr_object_obs = np.asarray(curr_obs, dtype=float).tolist()[17:20]
                curr_goal_obs = np.asarray(curr_obs, dtype=float).tolist()[20:]

                curr_frame = env.render()
                curr_img_pil = Image.fromarray(curr_frame)

                curr_img_name = f"ep{ep:04d}_step{step_count:03d}.png"
                curr_img_path = os.path.join(img_dir, curr_img_name)
                curr_img_pil.save(curr_img_path)

                record = {
                    "prev_image": img_name,
                    "curr_image": curr_img_name,
                    "episode": ep,
                    "step": step_count,
                    "reward": float(reward),
                    "prev_fingertip_obs": fingertip_obs,
                    "prev_object_obs": object_obs,
                    "prev_goal_obs": goal_obs,
                    "curr_fingertip_obs": curr_fingertip_obs,
                    "curr_object_obs": curr_object_obs,
                    "curr_goal_obs": curr_goal_obs,
                    "action": np.asarray(action, dtype=float).tolist(),
                    "goal_xy": new_goal_xy.tolist(),
                }

                f.write(json.dumps(record) + "\n")
                total_samples += 1

                prev_obs = curr_obs
                fingertip_obs = curr_fingertip_obs
                object_obs = curr_object_obs
                goal_obs = curr_goal_obs
                frame = curr_frame
                img_name = curr_img_name
                img_path = curr_img_path
                img_pil = curr_img_pil

    env.close()
    print(f"\nDataset collection complete! Saved {total_samples} samples to {save_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="pusher_vlm_data")
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()

    collect_pusher_dataset(
        save_dir=args.save_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )
