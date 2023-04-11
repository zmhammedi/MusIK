import torch
import torch.nn.functional as F
import numpy as np

class MusIK_Policy(object): 

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
    
    
    def act_batch(self, obs, I, t):
        # obs and I expected to be numpy.
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            I = torch.LongTensor(I).to(self.device)
            I_onehot = torch.squeeze(F.one_hot(I, num_classes=self.state_dim),dim=1).to(self.device).to(torch.float)

            tmp = self.learners[t].fs[0](obs).unsqueeze(1)
            for j in range(1,self.state_dim):
                tmp = torch.cat((tmp,self.learners[t].fs[j](obs).unsqueeze(1)),1)  
            
            IK_out = torch.matmul(I_onehot.unsqueeze(-2), tmp).squeeze(1)
           
            action = torch.argmax(IK_out, dim=1).unsqueeze(-1)
            #action = argmax // self.state_dim
            #J =  argmax % self.state_dim
            
        return action.cpu().data.numpy().flatten()#, J.cpu().data.numpy().flatten()
    
    
    def backward_index(self, I, t):
        # obs and I expected to be numpy.
        with torch.no_grad():
            I = torch.LongTensor(I).to(self.device)
            I_onehot = torch.squeeze(F.one_hot(I, num_classes=self.state_dim),dim=1).to(self.device).to(torch.float)
            J = torch.argmax(self.learners[t].fprim(I_onehot), dim=1).unsqueeze(-1) 
                        
        return J.cpu().data.numpy().flatten()
    
    
    def roll_in_musikpol(self, I, t, env):
        inds = []
        for l in range(t):
            inds.insert(0,I) 
            I = self.backward_index(I, t-l)
           
        ## Roll-in with the ith policy up to t-1
        obs = env.reset()
        for l in range(t):
            action = self.act_batch(obs, inds[l], l+1)  
            obs, _, _, _ = env.step(action) 
        
        return obs


    def eval_cover(self, h, eval_env, args, state_dim):
        layer_h_counts = 0 
        for i in range(state_dim):
            I = i * np.ones(args.num_eval, dtype=int)
            self.roll_in_musikpol(I, h, eval_env)
            layer_h_counts += eval_env.get_counts()  
            
        return layer_h_counts








