import numpy as np
import matplotlib.pyplot as plt
import os

def plot_eval_results(env_name, show=True, save=False):
    file_path = os.path.join('logs', env_name, 'evaluations.npz')
    
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    # Load data
    data = np.load(file_path)
    timesteps = data['timesteps']
    results = data['results']  # shape: (n_eval_runs, n_eval_episodes)

    # Compute mean and std dev
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    # Find best reward
    best_idx = mean_rewards.argmax()
    best_timestep = timesteps[best_idx]
    best_reward = mean_rewards[best_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label='Mean reward')
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='Std deviation')

    # Highlight best point
    plt.scatter(best_timestep, best_reward, color='red', zorder=5, label='Best reward')
    plt.annotate(f'Best: {best_reward:.2f}', (best_timestep, best_reward),
                 textcoords="offset points", xytext=(0, 10), ha='center', color='red')

    # Final touches
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title(f'Evaluation Reward Over Time - {env_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        save_dir = os.path.join('graphs')
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join('graphs', f'{env_name}.png')
        plt.savefig(save_path)

    plt.show()

if __name__ == '__main__':
    for env in ["Pendulum-v1", "BipedalWalker-v3", "HalfCheetah-v4", "Ant-v4"]:
        plot_eval_results(env, save=True)