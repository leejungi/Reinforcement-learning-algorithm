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

class PPO:
	def __init__(self,n_state, n_action, action_space=None,device='cpu'):
		#params
		self.n_state = n_state
		self.n_action = n_action
		self.device=device
		self.discount_factor = 0.99
		self.learning_rate = 3e-4 
		self.batch_size = 32 
		self.GAE = True
		self.train_epochs = 10
 
		self.model_update_interval = 1

		self.eps = torch.finfo(torch.float32).eps 
		
		
		#model define
		#action-value function
		self.Actor= Actor(self.n_state, self.n_action).to(self.device)		 
		self.Critic= Critic(self.n_state, 1).to(self.device)		 

		self.actor_optim = torch.optim.Adam(self.Actor.parameters(), lr=2e-5)
		self.critic_optim = torch.optim.Adam(self.Critic.parameters(), lr=3e-5)

		
	def get_action(self, state, evaluate=False):
		state = state.unsqueeze(0)
		with torch.no_grad():
			mean, log_std = self.Actor(state)
			std = log_std.exp()

			dist = Normal(mean, std)

			action = dist.sample()
		log_prob = dist.log_prob(action)
		action = torch.tanh(action)

#action = torch.tanh(action).detach().cpu().numpy()

		if evaluate == True:
			return torch.tanh(mean)

		return action.detach().cpu().numpy(), log_prob.item()

		   
	#Update
	def train(self, states, next_states, actions, rewards, masks, probs):
		next_states = torch.stack(next_states, dim=0).to(self.device)
		states = torch.stack(states, dim=0).to(self.device)
		actions = torch.FloatTensor(actions).to(self.device).view(-1,1)
		masks = torch.FloatTensor(masks).to(self.device).view(-1,1)

		old_prob = torch.FloatTensor(probs).to(self.device).view(-1,1)
		old_prob = old_prob.detach()

			
		if self.GAE == True:
			rewards = torch.FloatTensor(rewards).to(self.device).view(-1,1)

			with torch.no_grad():
				TD_target = rewards + self.discount_factor*self.Critic(next_states)*masks
			A =0
			Advantage =[]
			returns = (TD_target - self.Critic(states)).detach().cpu().numpy()

			for r in returns[::-1]:
				A = 0.95 * 0.99 * A + r[0]
				Advantage.append([A])
			Advantage.reverse()
			Advantage = torch.FloatTensor(Advantage).to(self.device)
		else:
#			returns = compute_returns(rewards)
#			TD_target= torch.FloatTensor(returns).to(self.device)
#			Advantage = (TD_target- self.Critic(states)).detach()
			rewards = torch.FloatTensor(rewards).to(self.device).view(-1,1)
			with torch.no_grad():
				TD_target = rewards + self.discount_factor*self.Critic(next_states)#*masks
				Advantage = (TD_target- self.Critic(states))


		for n in range(self.train_epochs):
			for index in BatchSampler(SubsetRandomSampler(range(len(rewards))), self.batch_size, False):
				mean, log_std = self.Actor(states[index])
				std = log_std.exp()

				dist = Normal(mean, std)
				present_prob = dist.log_prob(actions[index])
#			
#				with torch.no_grad():
#					TD_target = rewards[index] + self.discount_factor*self.Critic(next_states[index])#*masks
#					A = (TD_target- self.Critic(states[index]))
#				Q = TD_target.view(-1,1)

				Q = TD_target[index].view(-1,1)
				A = Advantage[index]
				

				ratio = torch.exp(present_prob - old_prob[index])
				ratio1 = ratio*A
				ratio2 = torch.clamp(ratio, 1.-0.2, 1.+0.2)*A
				actor_loss = -torch.min(ratio1, ratio2).mean()

				self.actor_optim.zero_grad()
				actor_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), 0.5)
				self.actor_optim.step()
			
				value = self.Critic(states[index])
				critic_loss = torch.nn.functional.mse_loss(Q, value)

				self.critic_optim.zero_grad()
				critic_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), 0.5)
				self.critic_optim.step()
				



			
def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
	def __init__(self,n_state, n_action):
		super(Actor, self).__init__()
		self.layers = nn.Sequential(
				nn.Linear(n_state, 64),
				nn.ReLU(),
				nn.Linear(64, 256),
				nn.ReLU(),
				)
		self.mean= nn.Linear(256, n_action)
		self.log_std= nn.Linear(256, n_action)
		self.apply(weights_init_)

		self.softplus = nn.Softplus()
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
				nn.Linear(n_state , 64),
				nn.ReLU(),
				nn.Linear(64, 256),
				nn.ReLU(),
				nn.Linear(256, n_action),
				
				)
		self.apply(weights_init_)

	def forward(self, state):		 
		return self.layers(state)



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

def compute_returns(rewards, gamma=0.99):
	R =0 
	returns = []
	for step in reversed(range(len(rewards))):
		R = rewards[step] + gamma*R
		returns.insert(0,R)
	return returns


if __name__ =="__main__":
	device= 'cuda' if torch.cuda.is_available() else 'cpu'
	
	print("Run on ", device)
	
	torch.manual_seed(777)
	if device == 'cuda':		
		torch.cuda.manual_seed_all(777)
	
#env = NormalizedActions(gym.make('Pendulum-v0'))
	env = gym.make('Pendulum-v0')
	save_path='PPO.pth'
	num_episode = 10000
	
	n_state = env.observation_space.shape[0]
	n_action = env.action_space.shape[0] 
	max_step = env._max_episode_steps

	policy = PPO(n_state, n_action, device=device)

	avg_reward = deque(maxlen=100)

	states, next_states, actions, rewards, masks, probs = [], [], [], [], [], []
	train_interval_ep = 4
	train_count =0

	#Training		 
	for i in range(num_episode):
		
		state = env.reset()
		state = np.reshape(state, [-1])    
		state = torch.Tensor(state).float().to(device)

		step = 0
		total_reward=0
		train_count +=1

		for t in range(max_step):  
			# env.render()

			states.append(state)
			action, action_prob = policy.get_action(state)
			
			next_state, reward, done, _ = env.step(2.*action)
			
			state = next_state
			state = torch.Tensor(state).float().to(device).squeeze(1)
			mask = 1 if t==max_step-1 else float(not done)

			next_states.append(state)
			actions.append(action)
			rewards.append((reward+8)/8)
#rewards.append(reward)
			probs.append(action_prob)
			masks.append(mask)

			total_reward += reward
			step +=1

			if done:
				break

			
				

		if train_count % train_interval_ep ==0:
			train_count =0
			policy.train(states, next_states, actions, rewards, masks, probs)
			states, next_states, actions, rewards, masks, probs = [], [], [], [], [], []

		avg_reward.append(total_reward/step)


		if i%10 ==0:	
			print("Episode: {} Rewards: {} Avg 100 rewards: {}".format(i, avg_reward[-1], np.mean(avg_reward)))
#print(f"Episode: {i} Rewards: {avg_reward[-1][0]:} Avg 100 rewards: {sum(avg_reward)/len(avg_reward):.4f}")
			
		
#	 policy.save_model(save_path)
	
#	plt.plot([i for i in range(len(step_list))],step_list)
#	plt.show()
	env.close()
