import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from collections import deque, namedtuple
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Replay_memory:
	def __init__(self, size):
		self.size = size
		self.buffer = []
		self.pos = 0

	def save_sample(self, state, action, reward, done, next_state):
		if len(self.buffer) < self.size:
			self.buffer.append(None)
		self.buffer[self.pos] = (state, action, reward, done, next_state)
		self.pos = (self.pos + 1) % self.size
	
	def get_sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, done, next_state = map(np.stack, zip(*batch))
		return state, action, reward, done, next_state 
	
	def __len__(self):
		return len(self.buffer)

class SAC:
	def __init__(self,n_state, n_action, action_space=None,device='cpu'):
		#params
		self.n_state = n_state
		self.n_action = n_action
		self.device=device
		self.action_space = action_space
		action_space=None
		if action_space == None:
			self.action_scale = torch.tensor(1.)
			self.action_bias= torch.tensor(0.)
		else:
			self.action_scale = torch.FloatTensor((self.action_space.high-self.action_space.low)/2.).to(self.device)
			self.action_bias= torch.FloatTensor((self.action_space.high-self.action_space.low)/2.).to(self.device)
		self.discount_factor = 0.99
		self.batch_size = 100 
 
		self.model_update_interval = 1
		self.memory_size = 1000000
		self.learning_rate = 0.0001 
		self.alpha = 0.9
		self.tau=0.005

		self.memory = Replay_memory(self.memory_size)
		self.eps = 1e-6
		
		
		#model define
		#action-value function
		self.Actor= Actor(self.n_state, self.n_action).to(self.device)		 
		self.Critic1= Critic(self.n_state, self.n_action).to(self.device)		 
		self.Critic2= Critic(self.n_state, self.n_action).to(self.device)		 

		self.target_Critic1= Critic(self.n_state, self.n_action).to(self.device)		 
		self.target_Critic2= Critic(self.n_state, self.n_action).to(self.device)		 

		self.actor_optim = torch.optim.Adam(self.Actor.parameters(), lr=self.learning_rate)
		self.critic1_optim = torch.optim.Adam(self.Critic1.parameters(), lr=self.learning_rate)
		self.critic2_optim = torch.optim.Adam(self.Critic2.parameters(), lr=self.learning_rate)
		
		
	def save_sample(self, state, action, reward, done, next_state):
		self.memory.save_sample(state, action, reward, done, next_state)

	def save_model(self, path):
		torch.save(self.model.state_dict(),path)
		
	def load_model(self, path):
		self.model.load_state_dict(torch.load(save_path))
			
	def get_action(self, state, evaluate=False):
		mean, log_std = self.Actor(state)
		std = torch.exp(log_std)
		dist = Normal(mean, std)
		action = 2.*torch.tanh(dist.sample()).detach().cpu().numpy()

		if evaluate == True:
			return torch.tanh(mean)
		return action
		   
	#Update
	def train(self):
		states, actions, rewards, dones, next_states = self.memory.get_sample(self.batch_size)

		states = torch.FloatTensor(states).to(self.device)
		next_states = torch.FloatTensor(next_states).to(self.device)
		actions = torch.LongTensor(actions).to(self.device)
		rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
		dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
		
#for indices in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
		#Soft Q value function update
		with torch.no_grad():
			mean, log_std = self.Actor(next_states)
			std = log_std.exp()
			normal = Normal(mean, std)
			resample = normal.rsample()

#			next_action = torch.tanh(resample) * 2.
#			next_log_pi = normal.log_prob(resample).sum(axis=-1)
#			next_log_pi -= (2*(np.log(2) + resample - F.softplus(2*resample))).sum(axis=1)
#			next_log_pi = next_log_pi.unsqueeze(1)

			next_action = torch.tanh(resample) * self.action_scale + self.action_bias
			next_log_pi = normal.log_prob(resample) - torch.log(self.action_scale*(1-next_action.pow(2))+ self.eps)
			next_log_pi = next_log_pi.sum(1, keepdim=True)

			#Q update: Q(st, at) - (r - gamma*V(st+1)) 
			next_Q1 = self.target_Critic1(next_states, next_action*2.)
			next_Q2 = self.target_Critic2(next_states, next_action*2.)

			#V(st+1) = Q(st,at) - alpha*logPi(at|st)
			next_V = torch.min(next_Q1, next_Q2) - self.alpha * next_log_pi
			
			TD_target = rewards + dones*self.discount_factor*next_V

		Q1 = self.Critic1(states, actions*2.)
		Q2 = self.Critic2(states, actions*2.)
		
		critic1_loss = F.mse_loss(TD_target,Q1)
		self.critic1_optim.zero_grad()
		critic1_loss.backward()
		self.critic1_optim.step()

		critic2_loss = F.mse_loss(TD_target,Q2)
		self.critic2_optim.zero_grad()
		critic2_loss.backward()
		self.critic2_optim.step()


		#Actor Update
		mean, log_std = self.Actor(states)
		std = log_std.exp()
		normal = Normal(mean, std)
		resample = normal.rsample()
		
#		pi_action = torch.tanh(resample) * 2.
#		log_pi = normal.log_prob(resample).sum(axis=-1)
#		log_pi -= (2*(np.log(2) + resample - F.softplus(2*resample))).sum(axis=1)
#		log_pi = log_pi.unsqueeze(1)
	
		pi_action = torch.tanh(resample)*self.action_scale + self.action_bias
		log_pi = normal.log_prob(resample) - torch.log(self.action_scale*(1-pi_action.pow(2))+ self.eps)
		log_pi = log_pi.sum(1, keepdim=True)

		Q1 = self.Critic1(states, pi_action*2.)
		Q2 = self.Critic1(states, pi_action*2.)
		
		#Actor update: mean(logPi(at|st) - Q(st,at))
		actor_loss = (self.alpha*log_pi - torch.min(Q1,Q2)).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()
	
	def sync(self):
		with torch.no_grad():
			for target_param, param in zip(self.target_Critic1.parameters(), self.Critic1.parameters()):
				target_param.data.copy_(target_param*(1-self.tau) + param.data*self.tau)

			for target_param, param in zip(self.target_Critic2.parameters(), self.Critic2.parameters()):
				target_param.data.copy_(target_param*(1-self.tau) + param.data*self.tau)

def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
	def __init__(self,n_state, n_action):
		super(Actor, self).__init__()
		self.layers = nn.Sequential(
				nn.Linear(n_state, 256),
				nn.ReLU(),
				nn.Linear(256,128),
				nn.ReLU(),
				nn.Linear(128,64),
				nn.ReLU()
				)
		self.mean= nn.Linear(64, n_action)
		self.log_std= nn.Linear(64, n_action)
		self.apply(weights_init_)

		self.log_min = -2
		self.log_max = 20

	def forward(self, x):		 
		x = self.layers(x)
		mean = self.mean(x)
		log_std = self.log_std(x)
		log_std = torch.clamp(log_std, min=self.log_min, max = self.log_max)

		return mean, log_std 
			
			
class Critic(nn.Module):
	def __init__(self,n_state, n_action):
		super(Critic, self).__init__()
		self.layers = nn.Sequential(
				nn.Linear(n_state + n_action, 256),
				nn.ReLU(),
				nn.Linear(256,128),
				nn.ReLU(),
				nn.Linear(128,64),
				nn.ReLU(),
				nn.Linear(64, 1),
				)
		self.apply(weights_init_)

	def forward(self, state, action):		 
		x = torch.cat([state, action],1)
		return self.layers(x)


		
class MLP(nn.Module):
	def __init__(self,n_state, n_action):
		super(MLP, self).__init__()
		self.layers = nn.Sequential(
				nn.Linear(n_state, 64),
				nn.ReLU()
				)
		self.actor = nn.Linear(64, n_action)
		self.critic = nn.Linear(64, 1)
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x):		 
		x = self.layers(x)

		actor_out = self.softmax(self.actor(x))
		critic_out = self.critic(x)
		return actor_out, critic_out
	
def test(policy, env, save_path=None,rendering=False):
	if save_path != None:
		policy.load_model(save_path)
	state = env.reset()
	state = np.reshape(state, [-1])		 
	max_step = env._max_episode_steps
	step =0
	with torch.no_grad():
		for t in range(max_step):
			if rendering == True:
				env.render()
			action = policy.get_action(state, True)
			next_state, reward, done, _ = env.step(action)
			
			step +=1
			state = next_state
			if done:			
				break
	return step

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action




if __name__ =="__main__":
	device= 'cuda' if torch.cuda.is_available() else 'cpu'
	
	print("Run on ", device)
	
	torch.manual_seed(777)
	if device == 'cuda':		
		torch.cuda.manual_seed_all(777)
	
#env = NormalizedActions(gym.make('Pendulum-v0'))
	env = gym.make('Pendulum-v0')
#	env._max_episode_steps = 10000
	save_path='SAC.pth'
	num_episode = 5000
	
	n_state = env.observation_space.shape[0]
	n_action = env.action_space.shape[0] 
	max_step = env._max_episode_steps
	max_step = 200


	policy = SAC(n_state, n_action, env.action_space, device=device)
	avg_reward = deque(maxlen=100)

	run_time =0

	#Training		 
	for i in range(num_episode):
		
		state = env.reset()
		state = np.reshape(state, [-1])    

		step = 0
		total_reward=0

		for t in range(max_step):  
			# env.render()

			if run_time > 10000:
				T_state = torch.Tensor(state).float().to(device)
				action = policy.get_action(T_state)*2.
			else:
				action = torch.FloatTensor(env.action_space.sample())
			next_state, reward, done, _ = env.step(action)
			
			mask = 1 if t==max_step-1 else float(not done)

			policy.save_sample(state, action, reward, mask, next_state )

			step +=1
			run_time +=1
			total_reward += reward
			
			state = next_state
				
			if done:
				break

		if run_time > 1000 and run_time % 50 ==0:
			for _ in range(50):
				policy.train()
				policy.sync()
		avg_reward.append(total_reward/step)


		
		print(f"Episode: {i} Rewards: {avg_reward[-1]:.4f} Avg 100 rewards: {sum(avg_reward)/len(avg_reward):.4f}")
			
		
#	 policy.save_model(save_path)
	
#	plt.plot([i for i in range(len(step_list))],step_list)
#	plt.show()
	env.close()
