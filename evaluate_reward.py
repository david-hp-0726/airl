import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt

from airl import RewardNet, make_env, device, gamma  # your AIRL definitions

def plot_results(stats_path, exp_rewards, rand_rewards, env_id):
    stats = np.load(stats_path)
    disc_loss   = stats["disc_loss"]
    pg_loss     = stats["pg_loss"]
    critic_loss = stats["critic_loss"]

    # Plot losses
    plt.figure()
    plt.plot(disc_loss,   label="Discriminator")
    plt.plot(pg_loss,     label="Policy")
    plt.plot(critic_loss, label="Critic")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{env_id} AIRL Training Losses")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join("graphs", f"{env_id}-losses.png")
    plt.savefig(save_path)

    plt.show()


def evaluate_reward(
    env_id: str,
    plot: bool,
    num_expert_eps: int = 10,
    num_random_eps: int = 10,
    max_steps: int = 2000,
):
    # ——————————————
    # 1) hard-coded paths
    base       = f"models/{env_id}"
    reward_path = f"airl_models/{env_id}/reward.pth"
    expert_zip  = f"{base}/best_model.zip"
    vec_path    = f"{base}/vec_normalize.pkl"
    stats_path  = f"airl_models/{env_id}/train_stats.npz"

    # ——————————————
    # 2) load AIRL reward net
    env_norm = make_env(env_id)
    obs_dim  = env_norm.observation_space.shape[0]
    act_dim  = env_norm.action_space.shape[0]

    reward_net = RewardNet(obs_dim, act_dim).to(device)
    reward_net.load_state_dict(torch.load(reward_path, map_location=device))
    reward_net.eval()

    def score(s, a, s_next):
        with torch.no_grad():
            return reward_net.f(
                torch.from_numpy(s).float().unsqueeze(0).to(device),
                torch.from_numpy(a).float().unsqueeze(0).to(device),
                torch.from_numpy(s_next).float().unsqueeze(0).to(device),
            ).item()

    # ——————————————
    # 3) load expert policy + VecNormalize
    expert_env = DummyVecEnv([lambda: gym.make(env_id)])
    expert_env = VecNormalize.load(vec_path, expert_env)
    expert_env.training    = False
    expert_env.norm_reward = False

    expert = SAC.load(expert_zip, env=expert_env)

    # ——————————————
    # 4) roll out expert policy
    exp_rewards = []
    for _ in range(num_expert_eps):
        obs = expert_env.reset()
        total = 0.0
        for _ in range(max_steps):
            action, _ = expert.predict(obs, deterministic=True)
            next_obs, _, done, _ = expert_env.step(action)
            total += score(obs, action, next_obs)
            obs = next_obs
            if done:
                break
        exp_rewards.append(total)

    print(env_id)
    print(f"Expert-policy: {np.mean(exp_rewards):.4f} ± {np.std(exp_rewards):.4f}")

    # ——————————————
    # 5) roll out random policy
    rand_rewards = []
    for _ in range(num_random_eps):
        obs = env_norm.reset()
        total = 0.0
        for _ in range(max_steps):
            action = env_norm.action_space.sample()
            action = np.array([action])
            next_obs, _, done, _ = env_norm.step(action)
            total += score(obs, action, next_obs)
            obs = next_obs
            if done:
                break
        rand_rewards.append(total)

    print(f"Random-policy: {np.mean(rand_rewards):.4f} ± {np.std(rand_rewards):.4f}")

    # ——————————————
    # 6) generate all plots
    if plot:
        plot_results(stats_path, exp_rewards, rand_rewards, env_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=False, help="Gym name, e.g. Pendulum-v1")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.env is not None:
        evaluate_reward(env_id=args.env, plot=args.plot)
    else:
        for env in ['Ant-v4', 'BipedalWalker-v3', 'HalfCheetah-v4', 'Pendulum-v1']:
            evaluate_reward(env_id=env, plot=args.plot)
