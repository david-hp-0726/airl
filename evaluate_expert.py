import argparse
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

def load_sac_model(env_name):
    model_dir = os.path.join("models", env_name)

    # Load and wrap environment
    env = DummyVecEnv([lambda: gym.make(env_name)])
    env = VecNormalize.load(os.path.join(model_dir, "vec_normalize.pkl"), env)
    env.training = False
    env.norm_reward = False

    # Load SAC model
    model = SAC.load(os.path.join(model_dir, "best_model.zip"), env=env)
    return model, env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Gym environment name (e.g. Ant-v4)")
    args = parser.parse_args()

    model, env = load_sac_model(args.env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"{args.env}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

if __name__ == "__main__":
    main()
