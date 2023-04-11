import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from algs.base_learner import weight_init, Encoder


        
class IK(nn.Module):
    def __init__(self, action_dim, device, obs_dim, state_dim=3, hidden_dim=100, tau=1, softmax="vanilla"):
        super().__init__()

        self.device = device
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.softmax = softmax
        
        self.ik = nn.Sequential(
            #nn.Linear(self.state_dim, self.hidden_dim), nn.Tanh(),
            #nn.Linear(self.obs_dim + self.state_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.obs_dim, self.hidden_dim), nn.Tanh(),
            #nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            #nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim * self.state_dim), nn.Tanh()
            #nn.Linear(self.hidden_dim, self.action_dim), nn.Tanh()
        )

        self.apply(weight_init)

    def forward(self, S):
        encoding = self.ik(S)
        if self.softmax == "gumble":
            ik = F.gumbel_softmax(encoding, tau=1, hard=True)
        elif self.softmax == 'vanilla': 
            ik = F.softmax(encoding / 10, dim=-1) #self.tau
        return ik

    def copy_ik(self, target):
        self.ik.load_state_dict(target.ik.state_dict())

    def save_ik(self, path):
        torch.save(self.ik.state_dict(), path)

    def load_ik(self, path):
        state_dict = torch.load(path)
        self.ik.load_state_dict(state_dict)



class MusIKLearner(object):
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
        tau_IK = 1, 
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
        self.phi = Encoder(obs_dim, action_dim, device, state_dim, tau=tau_encoder, softmax=softmax).to(device)
        self.fprim = IK(1, device, state_dim, state_dim, hidden_dim, tau=tau_IK, softmax=softmax).to(device)
        self.fs = []
        for i in range(self.state_dim):
            self.fs.append(IK(action_dim, device, obs_dim, 1, hidden_dim, tau=tau_IK, softmax=softmax).to(device))
        self.optimizer = optimizer
        self.reuse_weights = reuse_weights
        self.feature_lr = feature_lr
        self.feature_beta = feature_beta
        self.temp_path = temp_path

        self.fs_optimizer = []
        
        if self.optimizer == "Adam":
            self.phi_optimizer = torch.optim.Adam(
                self.phi.parameters(), lr=feature_lr, betas=(feature_beta, 0.999)
                )
            self.fprim_optimizer = torch.optim.Adam(
                self.fprim.parameters(), lr=feature_lr, betas=(feature_beta, 0.999)
                ) 
            
            for i in range(self.state_dim):
                self.fs_optimizer.append(torch.optim.Adam(
                    self.fs[i].parameters(), lr=feature_lr, betas=(feature_beta, 0.999))
                    )
        else:
            self.phi_optimizer = torch.optim.SGD(
                self.phi.parameters(), lr=feature_lr, momentum=0.99
            )
            self.fprim_optimizer = torch.optim.SGD(
                self.fprim.parameters(), lr=feature_lr, momentum=0.99
            ) 
            for i in range(self.state_dim):
                self.fs_optimizer.append(torch.optim.SGD(
                    self.fs[i].parameters(), lr=feature_lr, momentum=0.99) # feature_lr
            )


    def ik_learning(self, replay_buffer, h):
        reg_losses = []
        for m in range(self.num_feature_update):
            obs, actions, next_obs, _, _  = replay_buffer.sample(batch_size=self.batch_size)  # The output should already be a torch tensor
            actions=actions.to(torch.float)
            #actions = torch.tensor(actions, device=self.device).to(torch.float)
            ## I and actions already in one-hot format!            
            out = torch.zeros_like(actions, device=self.device)
            

            for j in range(self.state_dim):
                out += self.fs[j](obs) * self.phi(next_obs)[:,j].unsqueeze(-1)
            
            #reg = torch.mean(torch.sum(torch.sqrt(self.phi(obs)),1)).to(self.device)-1
            
            loss = - torch.mean(torch.log(torch.sum(actions * out,1))) 
            
           
            self.phi_optimizer.zero_grad()
            for j in range(self.state_dim):
                self.fs_optimizer[j].zero_grad()
            loss.backward()

            self.phi_optimizer.step() 
        
            for j in range(self.state_dim):
                self.fs_optimizer[j].step()
            
            reg_losses.append(loss.item())
            
        comp_losses = []
        for _ in range(self.num_feature_update):
            _, _, next_obs, _, i = replay_buffer.sample(batch_size=self.batch_size)  # The output should already be a torch tensor 
            #if t==h-1:
            #    i*=0
            # I and actions already in one-hot format!
            i_onehot = F.one_hot(i, num_classes=self.state_dim)
            i_onehot = torch.squeeze(i_onehot,dim=1).to(self.device) # Removing spurious dimension created by F.one_hot

            with torch.no_grad():
                phi = self.phi(next_obs)
                argmax = torch.argmax(phi, dim=1).unsqueeze(-1)
                S = F.one_hot(argmax, num_classes=self.state_dim)
                S = torch.squeeze(S, dim=1).to(torch.float).to(self.device)

            out = self.fprim(S) 
            
            #reg = torch.mean(torch.sum(torch.sqrt(self.phi(next_obs)),1)).to(self.device)-1
            loss_new = - torch.mean(torch.log(torch.sum(i_onehot * out,1))) #+self.lamb * reg
            
            #self.f_optimizer.zero_grad()
            self.fprim.zero_grad()
            loss_new.backward()
            #self.f_optimizer.step()
            self.fprim_optimizer.step()
            
            comp_losses.append(loss_new.item())
        
        return reg_losses, comp_losses
    

    def update(self, replay_buffer, h):

        reg_losses, comp_losses = self.ik_learning(replay_buffer, h)
      
        plt.plot(reg_losses)
        plt.savefig("{}/ik_loss_h={}.pdf".format(self.temp_path, str(h)))
        plt.close()
        
        plt.plot(comp_losses)
        plt.savefig("{}/comp_loss_h={}.pdf".format(self.temp_path, str(h)))
        plt.close()

        return np.mean(reg_losses)
    

    def save(self, h, H):
        self.phi.save_encoder("{}/phi_{}_{}.pth".format(self.temp_path,str(h),H))
        self.fprim.save_ik("{}/bk_{}_{}.pth".format(self.temp_path,str(h),H))
        for i in range(self.state_dim):
            self.fs[i].save_ik("{}/ik_{}_{}_{}.pth".format(self.temp_path,i,str(h),H))
    
    def load(self, h, H):
        self.phi.load_encoder("{}/phi_{}_{}.pth".format(self.temp_path,str(h),H))
        self.fprim.load_ik("{}/bk_{}_{}.pth".format(self.temp_path,str(h),H))
        for i in range(self.state_dim):
            self.fs[i].load_ik("{}/ik_{}_{}_{}.pth".format(self.temp_path,i,str(h),H))











