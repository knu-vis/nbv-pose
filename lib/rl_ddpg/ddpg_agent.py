import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .ddpg_model import Actor, Critic
from .replay_memory import ReplayMemory, Transition
import copy

class DDPGAgent:
    def __init__(self, state_dim, action_dim, args, device='cpu', option='both', weight_decay=0):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayMemory(args.replay_memory_size)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.action_range = args.action_range
        self.step = 0
        self.noise_epsilon = 1.0       
        self.noise_decay = 0.98        
        self.noise_decay_step = 10000  
        self.noise_minimum = 0.05      
        
        self.option = option    

        self.actor = Actor(state_dim, action_dim, self.action_range, option).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(state_dim, action_dim, option).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic, weight_decay=weight_decay)

                

    def select_action(self, state, noise=0.5):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state[0], state[5]).cpu().data.numpy()
        self.actor.train()
        action += noise * np.random.randn(self.action_dim) * self.action_range * self.noise_epsilon
        self.step += 1
        if self.step % self.noise_decay_step == 0:
            self.noise_epsilon = max(self.noise_epsilon * self.noise_decay, self.noise_minimum)
        
        return action

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        heatmap_batch = torch.cat(batch.heatmap).to(self.device)
        depthmap_batch = torch.cat(batch.depthmap).to(self.device)
        action_batch = torch.FloatTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_heatmap_batch = torch.cat(batch.next_heatmap).to(self.device)
        next_depthmap_batch = torch.cat(batch.next_depthmap).to(self.device)
        
        Q_vals = self.critic(heatmap_batch, depthmap_batch, action_batch)
        next_actions = self.actor_target(next_heatmap_batch, next_depthmap_batch)
        next_Q = self.critic_target(next_heatmap_batch, next_depthmap_batch, next_actions)
        Q_prime = reward_batch + self.gamma * next_Q
        critic_loss = F.mse_loss(Q_vals, Q_prime)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic(heatmap_batch, depthmap_batch, self.actor(heatmap_batch, depthmap_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.update_target(self.actor_target, self.actor, self.tau)
        self.update_target(self.critic_target, self.critic, self.tau)
        

    def update_target(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
