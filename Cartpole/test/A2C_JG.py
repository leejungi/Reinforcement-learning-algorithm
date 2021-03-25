import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from collections import deque
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import copy
from torch.distributions import Categorical
class DDQN:
	def __init__(self,n_state, n_action, device='cpu'):
		#params
		self.n_state = n_state
		self.n_action = n_action
		self.device=device
		self.discount_factor = 0.99
		self.learning_rate=0.01 #0.001
 
		self.num_step =0
		self.num_train = 0
		self.model_update_interval = 10

		self.eps = torch.finfo(torch.float32).eps
		
		
		#model define
		#action-value function
		self.model = MLP(self.n_state, self.n_action).to(self.device)		 
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
		
		
	def save_model(self, path):
		torch.save(self.model.state_dict(),path)
		
	def load_model(self, path):
		self.model.load_state_dict(torch.load(save_path))
			
	def get_action(self, state):
		state = state.unsqueeze(0)
		with torch.no_grad():
			dist, _ = self.model(state)
			dist = Categorical(dist)
			action = dist.sample()
			return action.item()
	   
	#Update
	def train(self, states, next_states, actions, rewards):

		#Compute Log probability
		next_states = torch.stack(next_states, dim=0).to(self.device)
		states = torch.stack(states, dim=0).to(self.device)
		
		dist, value = self.model(states)
		_, next_value = self.model(next_states)
		dist = Categorical(dist)			
		actions = torch.FloatTensor(actions).to(self.device)
		log_prob = dist.log_prob(actions)

		#Compute Advantage
		t_rewards = torch.FloatTensor(rewards).to(self.device)
		t_rewards = (t_rewards - t_rewards.mean()) / (t_rewards.std() + self.eps)
		with torch.no_grad():
			returns = t_rewards + self.discount_factor * next_value.detach()
#		returns = compute_returns(rewards) 
#		returns = torch.Tensor(returns).to(self.device)

		#Normalize
		returns = (returns - returns.mean()) / (returns.std() + self.eps)
		Advantage = returns - value

		Advantage = (Advantage - Advantage.mean()) /(Advantage.std() + self.eps)

		actor_loss = - (Advantage.detach() * log_prob).mean()
		critic_loss = Advantage.pow(2.).mean()

		entropy = dist.entropy().mean()

		loss = actor_loss + 0.5* critic_loss + 0.01*entropy	

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.6)
		self.optimizer.step()

		
		
class MLP(nn.Module):
	def __init__(self,n_state, n_action):
		super(MLP, self).__init__()
		self.layers = nn.Sequential(
				nn.Linear(n_state, 64),
				nn.ReLU()
				)
		self.actor = nn.Linear(64, n_action)
		self.critic = nn.Linear(64, 1)
		self.softmax = nn.Softmax(dim=1)

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

def compute_returns(rewards, gamma=0.99):
	R = 0
	returns = []
	for step in reversed(range(len(rewards))):
		R = rewards[step] + gamma * R 
		returns.insert(0, R)
	return returns



if __name__ =="__main__":
	device= 'cuda' if torch.cuda.is_available() else 'cpu'
	
	print("Run on ", device)
	
	torch.manual_seed(777)
	if device == 'cuda':		
		torch.cuda.manual_seed_all(777)
	
	
	env = gym.make('CartPole-v0')
	
	env._max_episode_steps = 10000
	save_path='DDQN.pth'
	num_episode = 5000
	
	
	n_state = env.observation_space.shape[0]
	n_action = env.action_space.n
	max_step = env._max_episode_steps


	policy = DDQN(n_state, n_action, device)
	avg_step =[]
	avg_reward =[]
	step_list = []

	states, next_states, actions, rewards = [], [], [], []

	#Training		 
	for i in range(num_episode):
		
		state = env.reset()
		state = np.reshape(state, [-1])    
		
		state = torch.Tensor(state).float().to(device)

		step = 0
		total_reward=0
		for t in range(max_step):  
			# env.render()
			

			states.append(state)
			action = policy.get_action(state)
			next_state, reward, done, _ = env.step(action)
			
			if done:
				reward = -100
			actions.append(action)	
			rewards.append(reward)
			
			step += 1

			total_reward += reward
			
			state = next_state
			state = torch.Tensor(state).float().to(device)
			next_states.append(state)
			
				
			if done:
				break

		policy.train(states, next_states, actions, rewards)

		avg_step.append(step)
		avg_reward.append(total_reward)
		step_list.append(step)

		states, next_states, actions, rewards = [], [], [], []

		
		print(f"Episode: {i} step: {step}")
		if (i+1)%100 == 0:
#			print(f"Episode: {i+1} Avg step: {np.mean(avg_step)} Avg reward: {np.mean(avg_reward)} ")
			avg_step =[]
			avg_reward =[]
#			 step = test(policy, env)
#			 print("Test step: ",step)
			
		if np.mean(step_list[-10:]) >= max_step:
			break
		
#	 policy.save_model(save_path)
	
	plt.plot([i for i in range(len(step_list))],step_list)
	plt.show()
	
	
	# Test
#	 step = test(policy, env, save_path,  True)
#	 print("Test step: ", step)

	env.close()
