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
    venv = DummyVecEnv([lambda: gym.make(env_id)])
    
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

    policy_rewards=[]

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

        policy_rewards.append(total_reward)

    mean = np.mean(policy_rewards)
    std = np.std(policy_rewards)

    print(f"Learned policy mean reward (over {num_episodes} episodes): {mean:.2f}, Std: {std:.2f}")

    random_rewards = []

    for _ in range(num_episodes):
        obs = venv.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = np.array([venv.action_space.sample()])
            obs, reward, done, _ = venv.step(action)
            total_reward += reward

        random_rewards.append(total_reward)
    
    mean = np.mean(random_rewards)
    std = np.std(random_rewards)

    print(f"Random policy mean reward (over {num_episodes} episodes): {mean:.2f}, Std: {std:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=False)
    args = parser.parse_args()

    envs = [args.env] if args.env is not None else ['Ant-v4', 'BipedalWalker-v3', 'HalfCheetah-v4', 'Pendulum-v1']

    for env in envs:
        policy_path = f'airl_models/{env}/policy.pth'
        vecnormalize_path = f'models/{env}/vec_normalize.pkl'

        evaluate_airl_policy(
            policy_path=policy_path,
            vecnormalize_path=vecnormalize_path,
            env_id=env,
            num_episodes=10
        )
   