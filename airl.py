import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hyperparams
gamma         = 0.99
disc_lr       = 5e-5
policy_lr     = 1e-3
critic_lr     = 1e-3
disc_batch    = 256
policy_batch  = 5000    # total timesteps per policy update; should be greater than the 
# num_rollouts_per_it = 5
traj_max_len  = 200     # Pendulum max episode length
hidden_dim = 256
disc_decay = 1e-4


def make_env(env_name):
        venv = DummyVecEnv([lambda: gym.make(env_name)])
        env = VecNormalize.load(f"models/{env_name}/vec_normalize.pkl", venv)
        # fix the normalization stats, don’t keep updating them
        env.training   = False
        env.norm_reward = False
        return env

class RewardNet(nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            # g(s,a)
            self.g = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),                   nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            # h(s)
            self.h = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),      nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def f(self, s, a, s_next):
            ga = self.g(torch.cat([s, a], dim=-1))                 # [B,1]
            hs = self.h(s)                                         # [B,1]
            hs_next = self.h(s_next)                               # [B,1]
            return ga + gamma * hs_next - hs                      # f_θ(s,a,s')
        
class PolicyNet(nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),      nn.Tanh(),
                nn.Linear(hidden_dim, action_dim)
            )
            # learnable log‐std
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, s):
            mu = self.net(s)
            std = torch.exp(self.log_std)
            return mu, std

        def get_dist(self, s):
            mu, std = self(s)
            return Normal(mu, std)

class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1) # (batch_size, )

def airl(env_name, n_iters=100):
    # 0) initialize training stats dict
    stats = {
        'pg_loss': [],
        'disc_loss': [],
        'critic_loss': []
    }

    # 1) load expert data (assumes you saved arrays 'obs','acts','next_obs')
    data = np.load(f"data/{env_name}/expert_data.npz")
    expert_obs      = torch.from_numpy(data["obs"     ]).float().to(device)
    expert_acts     = torch.from_numpy(data["acts"    ]).float().to(device)
    expert_next_obs = torch.from_numpy(data["next_obs"]).float().to(device)
    expert_dataset  = TensorDataset(expert_obs, expert_acts, expert_next_obs)
    expert_loader   = DataLoader(expert_dataset, batch_size=disc_batch, shuffle=True)

    # 2) make env with the same VecNormalize wrapper
    env = make_env(env_name)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 3) define policy network (REINFORCE) and value network
    policy = PolicyNet(obs_dim, action_dim).to(device)
    policy_opt = optim.Adam(policy.parameters(), lr=policy_lr)
    critic = ValueNet(obs_dim).to(device)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    # 4) define AIRL discriminator / reward network
    reward_net = RewardNet(obs_dim, action_dim).to(device)
    disc_opt    = optim.Adam(reward_net.parameters(), lr=disc_lr, weight_decay=disc_decay)
    bce_loss    = nn.BCEWithLogitsLoss()


    # helper: compute D(s,a,s') = sigmoid( f(s,a,s') - log π(a|s) )
    def discriminator(s, a, s_next):
        f_val = reward_net.f(s, a, s_next).squeeze(-1)      # [B]
        dist  = policy.get_dist(s)
        logp  = dist.log_prob(a).sum(-1)                   # [B]
        logits = f_val - logp
        return logits, torch.sigmoid(logits)

    # 5) training loop
    for it in range(1, n_iters+1):
        # ——— 5.1 collect policy data —— 
        policy_obs, policy_acts, policy_next, logps, dones = [], [], [], [], []
        while len(policy_obs) < policy_batch:
            obs = env.reset()
            done = False
            while not done:
                obs_tensor = torch.from_numpy(obs).float().to(device)
                dist = policy.get_dist(obs_tensor)
                action = dist.sample()
                next_obs, _, done, _ = env.step(action.cpu().numpy())
                policy_obs.append(obs_tensor)
                policy_acts.append(action)
                policy_next.append(torch.from_numpy(next_obs).float().to(device))
                logps.append(dist.log_prob(action).sum(-1))
                dones.append(torch.tensor(done, dtype=torch.float32, device=device))
                obs = next_obs if not done else env.reset()
                
                if len(policy_obs) >= policy_batch:
                     break
        policy_obs   = torch.cat(policy_obs) # (batch_size, obs_dim)
        policy_acts  = torch.cat(policy_acts)
        policy_next  = torch.cat(policy_next)
        policy_logp  = torch.cat(logps) # (batch_size, )
        dones = torch.cat(dones)
        # print(f"policy_obs: {policy_obs.shape}")
        # print(f"policy_act: {policy_acts.shape}")
        # print(f"policy_next: {policy_next.shape}")
        # print(f"policy_logp: {policy_logp.shape}")

        # ——— 5.2 train discriminator ——
        # we’ll take one pass over expert_loader and a matching
        # number of policy samples
        policy_dataset = TensorDataset(policy_obs, policy_acts, policy_next, policy_logp)
        policy_loader  = DataLoader(policy_dataset, batch_size=disc_batch, shuffle=True)

        reward_net.train()
        for (s_e, a_e, s_e_nxt), (s_p, a_p, s_p_nxt, lp_p) in zip(expert_loader, policy_loader):
            s_e, a_e, s_e_nxt = s_e.to(device), a_e.to(device), s_e_nxt.to(device)
            s_p, a_p, s_p_nxt = s_p.to(device), a_p.to(device), s_p_nxt.to(device)
            lp_p = lp_p.detach()
            lp_p = lp_p.to(device)

            # expert logits, policy logits
            logit_e, _ = discriminator(s_e, a_e, s_e_nxt)
            f_p = reward_net.f(s_p, a_p, s_p_nxt).squeeze(-1)
            logits_p = f_p - lp_p

            loss_e = bce_loss(logit_e, torch.ones_like(logit_e))
            loss_p = bce_loss(logits_p, torch.zeros_like(logits_p))
            loss = loss_e + loss_p

            # if loss.item() < 0.1:
            #     print("Skipping discriminator update")
            #     break
            disc_opt.zero_grad()
            loss.backward()
            disc_opt.step()

        # ——— 5.3 compute AIRL reward for policy data ——
        with torch.no_grad():
            logit_p, Dp = discriminator(policy_obs, policy_acts, policy_next)
            # r = log D − log (1−D) = logit_p
            rewards = (logit_p - logit_p.mean()) / (logit_p.std() + 1e-8)
            rewards = rewards.detach()

        # ——— 5.4 update policy via REINFORCE ——
        # compute discounted returns
        # Update critic
        with torch.no_grad():
             next_values = critic(policy_next)
             td_target = rewards + (1 - dones) * gamma * next_values
        values = critic(policy_obs)

        critic_opt.zero_grad()
        critic_loss = F.mse_loss(values, td_target)
        critic_loss.backward()
        critic_opt.step()

        # Update policy
        with torch.no_grad():
             adv = td_target - critic(policy_obs)
             adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # advantages *= 10 # Increase policy gradient scale
        policy_opt.zero_grad()
        # loss = −E[ return · log π(a|s) ]
        if it % 5 == 0:
            print(f"values mean = {values.mean()} adv mean = {adv.mean()} logp mean = {policy_logp.mean()}")
        pg_loss = - (adv * policy_logp).mean()
        pg_loss.backward()
        policy_opt.step()


        print(f"[Iter {it}/{n_iters}] Disc loss {loss.item():.3f}   PG loss {pg_loss.item():.3f}    Critic loss {critic_loss.item():.3f}")
        stats['disc_loss'].append(loss.item())
        stats['pg_loss'].append(pg_loss.item())
        stats['critic_loss'].append(critic_loss.item())


    # save final policy & reward networks
    save_dir = os.path.join('airl_models', env_name)
    os.makedirs(save_dir, exist_ok=True)

    policy_path = os.path.join(save_dir, "policy.pth")
    reward_path = os.path.join(save_dir, "reward.pth")
    torch.save(policy.state_dict(),      policy_path)
    torch.save(reward_net.state_dict(),   reward_path)

    # save stats
    stats_path = os.path.join(save_dir, "train_stats.npz")
    np.savez(stats_path, **{k: np.array(v) for k,v in stats.items()})
    print(f"Saved training stats to {stats_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=False, help="Gym environment name (e.g. Ant-v4)")
    parser.add_argument("--nit", type=int, required=False, help="#Training iterations")
    args = parser.parse_args()

    n_iters = args.nit if args.nit is not None else 100
    if args.env is not None:
        airl(args.env , n_iters=n_iters)
    else:
        for env in ['Ant-v4', 'BipedalWalker-v3', 'HalfCheetah-v4', 'Pendulum-v1']:
             airl(env, n_iters=n_iters)