import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = str(Path(__file__).parent.parent.parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from jax_guam.rl.guam_env import GuamEnv

def make_env():
    """Create and wrap the environment"""
    env = GuamEnv()
    env = Monitor(env)
    return env

def train():
    # Force CPU usage
    device = torch.device("cpu")
    
    # Create output directory
    log_dir = "ppo_guam_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create and wrap the environment
    env = DummyVecEnv([make_env])
    
    # Initialize the agent with modified hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # Reduced learning rate for stability
        n_steps=2048,        # Increased steps per update
        batch_size=64,
        n_epochs=10,         # Increased epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,      # Removed target KL to allow more exploration
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],  # Larger policy network
                vf=[256, 256]   # Larger value network
            )
        ),
        verbose=1,
        device=device
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,     # Save less frequently
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_guam"
    )
    
    # Train the agent
    total_timesteps = 100000  # Train for longer
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save(os.path.join(log_dir, "final_model"))
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,  # More evaluation episodes
        deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    train() 