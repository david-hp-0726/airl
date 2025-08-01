import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

def make_vec_env(vec_path: str, env_name: str):
    """
    Create a VecNormalize-wrapped Pendulum-v1 environment
    using the saved statistics at `vec_path`.
    """
    venv = DummyVecEnv([lambda: gym.make(env_name)])
    env = VecNormalize.load(vec_path, venv)
    env.training = False
    env.norm_reward = False
    return env


def collect_expert_data(
    env_name: str,
    n_episodes: int = 100,
    output_path: str = "expert_pendulum.npz",
    verbose: bool = False
):
    """
    Runs `n_episodes` of the expert policy and collects transitions:
    (state, action, next_state). Saves them in a .npz file.
    """
    model_path = f"models/{env_name}/best_model.zip"
    vec_path = f"models/{env_name}/vec_normalize.pkl"

    # load env and expert
    env = make_vec_env(vec_path, env_name)
    model = SAC.load(model_path, env=env)

    obs_buf, act_buf, next_obs_buf = [], [], []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, _ = env.step(action)

            obs_buf.append(obs.copy())
            act_buf.append(action.copy())
            next_obs_buf.append(next_obs.copy())

            obs = next_obs
            total_reward += reward
        if verbose:
            print(f"Ep {ep} finished with return {total_reward}")

    # convert to arrays and save
    obs_arr      = np.array(obs_buf)
    act_arr      = np.array(act_buf)
    next_obs_arr = np.array(next_obs_buf)

    np.savez(output_path,
             obs=obs_arr,
             acts=act_arr,
             next_obs=next_obs_arr)

    print(f"Saved {obs_arr.shape[0]} transitions to '{output_path}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect expert trajectories for AIRL (Pendulum-v1)"
    )
    parser.add_argument(
        "--env", type=str, help="name of environment"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="number of expert episodes to roll out"
    )
    parser.add_argument(
        "--output", type=str, default="expert_pendulum.npz",
        help="file path to save collected transitions"
    )
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    collect_expert_data(
        env_name=args.env,
        n_episodes=args.episodes,
        output_path=args.output,
        verbose=args.verbose
    )
