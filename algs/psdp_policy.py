#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 00:21:26 2023

@author: zmhammedi
"""

import torch

class PSDP_Policy(object): 

    def __init__(
        self,
        obs_dim,
        state_dim,
        action_dim,
        horizon,
        device,
        learners,
        lamb = 1,
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.feature_dim = state_dim
        self.device = device
        self.learners = learners
    
    
    def act_batch(self, obs, t):
        # obs and I expected to be numpy.
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action = torch.argmax(self.learners[t].f(obs), dim=1).unsqueeze(-1) 
            
        return action.cpu().data.numpy().flatten()
                

    def save_weight(self, path): 
        for h in range(self.horizon):
            torch.save(self.W[h],"{}/W_{}.pth".format(path,str(h)))
            torch.save(self.Sigma_invs[h], "{}/Sigma_{}.pth".format(path,str(h)))

    def load_weight(self, path):
        for h in range(self.horizon):
            self.W[h] = torch.load("{}/W_{}.pth".format(path,str(h)))
            self.Sigma_invs[h] = torch.load("{}/Sigma_{}.pth".format(path,str(h)))
