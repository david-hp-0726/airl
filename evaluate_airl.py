import torch
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from airl import PolicyNet
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_airl_policy(policy_path, vecnormalize_path, env_id, num_episodes=10):
    # Step 1: Load environment
    def make_env():
        return gym.make(env_id)

    venv = DummyVecEnv([make_env])
    
    # Step 2: Load VecNormalize
    venv = VecNormalize.load(vecnormalize_path, venv)
    venv.training = False
    venv.norm_reward = False  # Optional: don't normalize reward during evaluation

    # Step 3: Load trained policy
    obs_dim    = venv.observation_space.shape[0]
    action_dim = venv.action_space.shape[0]
    policy = PolicyNet(obs_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    episode_rewards=[]

    for _ in range(num_episodes):
        obs = venv.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = policy(obs_tensor)[0].numpy()
            obs, reward, done, _ = venv.step(action)
            total_reward += reward[0]
            if render:
                venv.render()

        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Evaluated over {num_episodes} episodes — Mean reward: {mean_reward:.2f}, Std: {std_reward:.2f}")
    return mean_reward, std_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    args = parser.parse_args

    policy_path = f'airl_models/{args.env}/policy.pth'
    vecnormalize_path = f'models/{args.env}/vec_normalize.pkl'

    evaluate_airl_policy(
        policy_path=policy_path,
        vecnormalize_path=vecnormalize_path,
        env_id=args.env,
        num_episodes=10,
        render=False
    )
