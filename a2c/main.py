import argparse
import os

import gym_map
import pygame
import gym
import time
import pickle
import numpy as np

from a2c.model import ActorCritic
from a2c.multiprocessing_env import SubprocVecEnv, VecPyTorch, VecPyTorchFrameStack
from a2c.wrappers import *
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, name):
    ofile = open("model_{}".format(name), "wb")
    pickle.dump(model, ofile)
    ofile.close()
    
def load_model(name):
    ifile = open("model_{}".format(name), "rb")
    model = pickle.load(ifile)
    ifile.close()
    return model
    

def parse_args():
    parser = argparse.ArgumentParser("args")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env", type=str, default="Map-v0")
    parser.add_argument("--actor-loss-coefficient", type=float, default=1.0)
    parser.add_argument("--critic-loss-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-loss-coefficient", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=5) # rollout
    parser.add_argument("--num-envs", type=int, default=2) # multiprocessing
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-frames", type=int, default=320000) # num timesteps to run
    return parser.parse_args()


def compute_returns(next_value, rewards, masks, gamma):
    r = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        r = rewards[step] + gamma * r * masks[step]
        returns.insert(0, r)
    return returns


def make_env(seed, rank):
    def _thunk():
        env = gym.make("Map-v0")
        env.reset()
        env.seed(seed + rank)

        allow_early_resets = False
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env
    return _thunk


def make_envs():
    envs = [make_env(args.seed, i) for i in range(args.num_envs)]
    envs = SubprocVecEnv(envs)
    envs = VecPyTorch(envs, device)
    envs = VecPyTorchFrameStack(envs, 4, device)
    return envs


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    envs = make_envs()

    actor_critic = ActorCritic(envs.observation_space, envs.action_space).to(device)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr=args.lr, eps=args.eps, alpha=args.alpha)

    num_updates = args.num_frames // args.num_steps // args.num_envs

    observation = envs.reset()
    start = time.time()

    episode_rewards = deque(maxlen=10)
    for update in range(num_updates):

        log_probs = []
        values = []
        rewards = []
        actions = []
        masks = []
        entropies = []

        for step in range(args.num_steps):
            observation = observation.to(device) / 255.
            actor, value = actor_critic(observation)

            action = actor.sample()
            next_observation, reward, done, infos = envs.step(action.unsqueeze(1))

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            log_prob = actor.log_prob(action)
            entropy = actor.entropy()

            mask = torch.from_numpy(1.0 - done).to(device).float()

            entropies.append(actor.entropy())
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward.to(device).squeeze())
            masks.append(mask)

            observation = next_observation

        next_observation = next_observation.to(device).float() / 255.
        with torch.no_grad():
            _, next_values = actor_critic(next_observation)
            returns = compute_returns(next_values.squeeze(), rewards, masks, args.gamma)
            returns = torch.cat(returns)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        entropies = torch.cat(entropies)

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()

        loss = args.actor_loss_coefficient * actor_loss + \
               args.critic_loss_coefficient * critic_loss - \
               args.entropy_loss_coefficient * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
        optimizer.step()
        
        if update % 10000 == 0:
            save_model(actor_critic, update)

    save_model(actor_critic)