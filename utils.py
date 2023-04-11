import argparse
import torch
import numpy as np

import random
import os

from envs.Lock_batch import LockBatch
from algs.musik_learner import MusIKLearner
from algs.psdp_learner import PSDPLearner

#seed_0_horizon_100_episodes_1100000_hidden_dim_100_updates_128_batch_size_1024

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', default="test", type=str)
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--temp_path', default="tmp", type=str)
    parser.add_argument('--load', default=False, type=bool)
    parser.add_argument('--dense', default=False, type=bool)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_episodes', default=60000, type=int) # 1010000 for H=100, 300000 for H=50, 65000 for H=25  
    parser.add_argument('--num_episodes_psdp', default=8000, type=int) # 200000 for H=100,  20000 for H=50, 8000 for H=25
    parser.add_argument('--batch_size', default=4096, type=int)

    #environment
    parser.add_argument('--horizon', default=25, type=int)
    parser.add_argument('--switch_prob', default=0.5, type=float)
    parser.add_argument('--anti_reward', default=0.1, type=float)
    parser.add_argument('--anti_reward_prob', default=0.5, type=float)
    parser.add_argument('--num_actions', default=10, type=int)
    parser.add_argument('--state_dim', default=3, type=int)
    parser.add_argument('--observation_noise', default=0.1, type=float)
    parser.add_argument('--variable_latent', default=False, type=bool)
    parser.add_argument('--env_temperature', default=0.2, type=float)

    #rep
    parser.add_argument('--rep_num_feature_update', default=128, type=int)
    parser.add_argument('--rep_num_feature_update_psdp', default=128, type=int)
    parser.add_argument('--feature_lr', default=.001, type=float)
    parser.add_argument('--feature_beta', default=0.9, type=float)
    parser.add_argument('--linear_beta', default=0.9, type=float)
    parser.add_argument('--rep_lamb', default=.0, type=float)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--temperature_encoder', default=1, type=float)
    parser.add_argument('--temperature_IK', default=1, type=float)

    parser.add_argument('--reuse_weights', default=True, type=bool)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--softmax', default='vanilla', type=str)

    #eval
    parser.add_argument('--num_eval', default=50, type=int)

    args = parser.parse_args()
    return args


def make_musik_learner(env, state_dim, device, args):
    musik_learners = []
    for h in range(args.horizon):
        musik_learners.append(MusIKLearner(env.observation_space.shape[0],
                 state_dim, 
                 env.action_dim,
                 args.hidden_dim,
                 args.rep_num_feature_update,
                 device,
                 feature_lr=args.feature_lr,
                 feature_beta=args.feature_beta,
                 batch_size = args.batch_size,
                 lamb = args.rep_lamb,
                 tau_encoder = args.temperature_encoder,
                 tau_IK = args.temperature_IK,
                 optimizer = args.optimizer,
                 softmax = args.softmax,
                 reuse_weights = args.reuse_weights,
                 temp_path = args.temp_path))
        
    return musik_learners   


def make_psdp_learner(env, state_dim, device, args):
    psdp_learners = []
    for h in range(args.horizon):
        psdp_learners.append(PSDPLearner(env.observation_space.shape[0],
                 state_dim,
                 env.action_dim,
                 args.hidden_dim,
                 args.rep_num_feature_update_psdp,
                 device,
                 feature_lr=args.feature_lr,
                 feature_beta=args.feature_beta,
                 batch_size = args.batch_size,
                 lamb = args.rep_lamb,
                 tau_encoder = args.temperature_encoder,
                 optimizer = args.optimizer,
                 softmax = args.softmax,
                 reuse_weights = args.reuse_weights,
                 temp_path = args.temp_path))
        
    return psdp_learners  


def make_batch_env(args, num_env):
    env = LockBatch()
    env.init(horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=num_env,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env = LockBatch()
    eval_env.init(horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=args.num_eval,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense)

    eval_env.seed(args.seed)
    eval_env.opt_a = env.opt_a
    eval_env.opt_b = env.opt_b

    return env, eval_env


def set_seed_everywhere(seed):
    torch.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, num_actions, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.num_actions = num_actions

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, num_actions), dtype=np.int)
        self.target_states = np.empty((capacity, 1), dtype=np.int)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False


    def add_batch(self, obs, action, next_obs, size, rewards=None, target_states=None):
        np.copyto(self.obses[self.idx:self.idx+size], obs)
        aoh = np.zeros((size,self.num_actions), dtype=np.int)
        aoh[np.arange(size), action] = 1
        np.copyto(self.actions[self.idx:self.idx+size], aoh)
        np.copyto(self.next_obses[self.idx:self.idx+size], next_obs)
        
        if rewards is not None:
            np.copyto(self.rewards[self.idx:self.idx+size], rewards)
            
        if target_states is not None:
            np.copyto(self.target_states[self.idx:self.idx+size], target_states)

        if self.idx + size >= self.capacity:
            self.full  = True
            self.idx = 0
        else:
            self.idx = self.idx + size
        #self.idx = (self.idx + size) % self.capacity
        #self.full = self.full or self.idx == 0

    def add_from_buffer(self, buf, h): 
        obs, action, target_states, next_obs = buf.get(h)
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.target_states[self.idx], target_states)
        np.copyto(self.next_obses[self.idx], next_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        target_states = torch.as_tensor(
            self.target_states[idxs], dtype=torch.long, device=self.device)
        
        return obses, actions, next_obses, rewards, target_states

        
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.target_states[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.target_states[start:end] = payload[3]
            self.idx = end
