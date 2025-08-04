from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import gymnasium as gym

class EvalCallbackWithVecNorm(EvalCallback):
    def __init__(self, *args, save_vecnormalize_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_vecnormalize_path = save_vecnormalize_path
        self.original_training = None
        self.original_norm_reward = None

    def _on_step(self) -> bool:
        # Toggle env attributes
        if hasattr(self.eval_env, "training"):
            self.original_training = self.eval_env.training
        if hasattr(self.eval_env, "norm_reward"):
            self.original_norm_reward = self.eval_env.norm_reward
        
        result = super()._on_step()

        if hasattr(self.eval_env, "training"):
            self.eval_env.training = self.original_training
        if hasattr(self.eval_env, "norm_reward"):
            self.eval_env.norm_reward = self.original_norm_reward

        # Save vec-normalized environment if best mean reward is achieved
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward == self.best_mean_reward:
                # New best model detected
                self.eval_env.save(os.path.join(self.save_vecnormalize_path, "vec_normalize.pkl"))
                if self.verbose == 1:
                    print("Normalized env saved to ", self.save_vecnormalize_path)
        return result

def train_expert(env_name, total_steps, log_dir='logs', model_dir='models'):
    # make directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(log_dir, env_name)
    model_path = os.path.join(model_dir, env_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # create vectorized envs
    env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
    env = VecNormalize(env)

    # define callback
    eval_callback = EvalCallbackWithVecNorm(
        eval_freq=10_000,
        eval_env=env,
        log_path=log_path,
        best_model_save_path=model_path,
        save_vecnormalize_path=model_path,
        n_eval_episodes=10
    )

    # train model
    sac = SAC(
        policy='MlpPolicy', 
        env=env,
        verbose=1,
        learning_starts=10_000
    )
    sac.learn(total_steps, callback=[eval_callback])


if __name__ == '__main__':
    env_names = ['Pendulum-v1', 'BipedalWalker-v3', 'HalfCheetah-v4', 'Ant-v4']
    total_steps = [500_000, 500_000, 1_000_000, 2_000_000]
    for i in range(0, len(env_names)):
        train_expert(env_name=env_names[i], total_steps=total_steps[i])