import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt

from algs.base_learner import weight_init


# TODO check that not saving over the geni pols
        
class LSS(nn.Module):
    def __init__(self, action_dim, device, obs_dim, state_dim=3, hidden_dim=100):
        super().__init__()

        self.device = device
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.lss = nn.Sequential(
            #nn.Linear(self.state_dim, self.hidden_dim), nn.Tanh(),
            #nn.Linear(self.obs_dim + self.state_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.obs_dim, self.hidden_dim), nn.Tanh(),
            #nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            #nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim), nn.Tanh()
            #nn.Linear(self.hidden_dim, self.action_dim), nn.Tanh()
        )

        self.apply(weight_init)

    def forward(self, obs):
        r = self.lss(obs) 
        
        return r

    def copy_lss(self, target):
        self.lss.load_state_dict(target.lss.state_dict())

    def save_lss(self, path):
        torch.save(self.lss.state_dict(), path)

    def load_lss(self, path):
        state_dict = torch.load(path)
        self.lss.load_state_dict(state_dict)



class PSDPLearner(object):
    def __init__(
        self,
        obs_dim,
        state_dim,
        action_dim,
        hidden_dim,
        rep_num_feature_update,
        device,
        feature_lr=1e-3,
        feature_beta=0.9,
        batch_size = 128,
        lamb = 1,
        tau_encoder = 1,
        optimizer = "sgd",
        softmax = "gumble",
        reuse_weights = True,
        temp_path = "temp"
    ):

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_feature_update = rep_num_feature_update
        self.feature_dim = state_dim * action_dim
        self.device = device
        self.lamb = lamb
        self.batch_size = batch_size
        
        
        self.f = LSS(action_dim, device, obs_dim, 1, hidden_dim).to(device)
        self.optimizer = optimizer
        self.reuse_weights = reuse_weights
        self.feature_lr = feature_lr
        self.feature_beta = feature_beta
        self.temp_path = temp_path

        
        if self.optimizer == "Adam":
            self.f_optimizer = torch.optim.Adam(
                    self.f.parameters(), lr=feature_lr, betas=(feature_beta, 0.999)) 

        else:
            self.f_optimizer = torch.optim.SGD(
                    self.f.parameters(), lr=feature_lr, momentum=0.99) 


    def ls_learning(self, replay_buffer, h):
        reg_losses = []
        for m in range(self.num_feature_update):
            obs, actions, _, R, _ = replay_buffer.sample(batch_size=self.batch_size)  # The output should already be a torch tensor
            actions = actions.to(torch.float)
                
            loss = F.mse_loss(torch.sum(actions * self.f(obs),1).unsqueeze(1), R) 
        
            self.f_optimizer.zero_grad()
            loss.backward()
            self.f_optimizer.step() 
        
            reg_losses.append(loss.item())
        
        return reg_losses
    

    def update(self, replay_buffer, h, phi=None):
    
        reg_losses = self.ls_learning(replay_buffer, h)
      
        plt.plot(reg_losses)
        plt.savefig("{}/lss_loss_h={}.pdf".format(self.temp_path, str(h)))
        plt.close()

        return np.mean(reg_losses)
    
    def save(self, h, H=0):
        self.f.save_lss("{}/ls_{}_{}.pth".format(self.temp_path,str(h),H))
        
    def load(self, h, H=0):
        self.f.load_lss("{}/ls_{}_{}.pth".format(self.temp_path,str(h),H))
        
        
    








