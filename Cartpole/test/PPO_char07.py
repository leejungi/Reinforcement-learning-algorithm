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
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
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
		
		self.buffer =[] 
		
		#model define
		#action-value function
		self.Actor= Actor(self.n_state, self.n_action).to(self.device)		 
		self.Critic= Critic(self.n_state, 1).to(self.device)		 
		self.actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=1e-3)
		self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=3e-3)
		
		
	def save_model(self, path):
		torch.save(self.model.state_dict(),path)
		
	def load_model(self, path):
		self.model.load_state_dict(torch.load(save_path))
			
	def get_action(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
		with torch.no_grad():
			prob= self.Actor(state)
		dist = Categorical(prob)
		action = dist.sample()
		return action.item(), prob[:,action.item()].item()
	def store_transition(self, transition):
		self.buffer.append(transition)

	#Update
	def train(self):

		epochs = 10
		
		state = torch.FloatTensor([t.state for t in self.buffer]).to(self.device)
		action = torch.LongTensor([t.action for t in self.buffer]).view(-1,1).to(self.device)

		reward = [t.reward for t in self.buffer]
		old_action_log_prob = torch.FloatTensor([t.a_log_prob for t in self.buffer]).view(-1,1).to(self.device)


		R = 0
		Gt =[]
		for r in reward[::-1]:
			R = r + 0.99*R
			Gt.insert(0,R)
		Gt = torch.FloatTensor(Gt).to(self.device)


		for n in range(epochs):
			for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), 32, False):
				Gt_index = Gt[index].view(-1,1)
				V = self.Critic(state[index])
				delta = Gt_index -V
				advantage = delta.detach()

				action_prob = self.Actor(state[index]).gather(1, action[index])

				ratio = (action_prob/old_action_log_prob[index])
				surr1 = ratio*advantage
				surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantage

				actor_loss = -torch.min(surr1, surr2).mean()

				
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), 0.5)
				self.actor_optimizer.step()
				
				critic_loss = torch.nn.functional.mse_loss(Gt_index, V)					

				self.critic_optimizer.zero_grad()
				critic_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), 0.5)
				self.critic_optimizer.step()
		del self.buffer[:]

class Actor(nn.Module):
	def __init__(self,n_state, n_action):
		super(Actor, self).__init__()
		self.layers = nn.Sequential(
				nn.Linear(n_state, 100),
				nn.ReLU()
				)
		self.actor = nn.Linear(100, n_action)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):		 
		x = self.layers(x)

		actor_out = self.softmax(self.actor(x))
		return actor_out
			
			
class Critic(nn.Module):
	def __init__(self,n_state, n_action):
		super(Critic, self).__init__()
		self.layers = nn.Sequential(
				nn.Linear(n_state, 100),
				nn.ReLU()
				)
		self.critic = nn.Linear(100, 1)

	def forward(self, x):		 
		x = self.layers(x)

		critic_out = self.critic(x)
		return critic_out

		
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
	
	seed = 1
	torch.manual_seed(seed)
	if device == 'cuda':		
		torch.cuda.manual_seed_all(seed)
	
	env = gym.make('CartPole-v0').unwrapped
	env.seed(seed)

	env._max_episode_steps = 10000
	save_path='DDQN.pth'
	num_episode = 1000
	batch_size = 32
	
	
	n_state = env.observation_space.shape[0]
	n_action = env.action_space.n
	max_step = env._max_episode_steps


	policy = DDQN(n_state, n_action, device)
	avg_step =[]
	avg_reward =[]
	step_list = []

	Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
	states, actions, rewards, probs = [], [], [], []
	train_count =0
	#Training		 
	for i in range(num_episode):
		
		state = env.reset()
		

		step = 0
		total_reward=0
		for t in range(max_step):  
			# env.render()
			

			action, action_prob = policy.get_action(state)
			next_state, reward, done, _ = env.step(action)

			trans = Transition(state, action, action_prob, reward, next_state)
			
			policy.store_transition(trans)
			
			step += 1

			total_reward += reward
			
			state = next_state
				
			if done:
				if len(policy.buffer) >=32: 
					policy.train()
				break

			states, actions, rewards, probs = [], [], [], []
		avg_step.append(step)
		avg_reward.append(total_reward)
		step_list.append(step)


		
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
