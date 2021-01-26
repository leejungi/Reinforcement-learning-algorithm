import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from collections import deque
import numpy as np
import random
import gym

class DQN:
    def __init__(self,n_state, n_action, device='cpu'):
        #params
        self.n_state = n_state
        self.n_action = n_action
        self.device=device
        self.learning_rate = 0.0001
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.discount_factor = 0.99
        self.learning_rate=0.001
        self.batch_size= 64
        
        self.train_start = 1000
        self.replay_memory_size = 50000#2000
        self.replay_memory = deque(maxlen=self.replay_memory_size)        
        
        #model define
        self.model = MLP(self.n_state, self.n_action).to(self.device)        
        self.loss = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def save_sample(self, sample):
        #sample = [state, action, reward, next_state, done]
        self.replay_memory.append(sample)

        
	#E-greedy in state
    def get_action(self, state,episode):

        if np.random.rand() <= self.epsilon:    
            action = random.randrange(self.n_action)
        else:
            # T means torch.Tensor
            T_state = torch.FloatTensor(state).to(self.device)
            T_q = self.model(T_state) 
            action = np.argmax(T_q.to('cpu').detach().numpy())        
            
        return action
    
    #Update
    def train(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.replay_memory, self.batch_size)
        mini_batch = np.array(mini_batch)
        

        # T_states = np.zeros((self.batch_size,self.n_state))
        # T_actions = np.zeros((self.batch_size))
        # T_rewards = np.zeros((self.batch_size))
        # T_next_states = np.zeros((self.batch_size,self.n_state))
        # T_dones = np.zeros((self.batch_size))

        # for i in range(self.batch_size):
        #     T_states[i] = mini_batch[i][0]
        #     T_actions[i] = mini_batch[i][1]
        #     T_rewards[i] = mini_batch[i][2]
        #     T_next_states[i] = mini_batch[i][3]
        #     T_dones[i] = mini_batch[i][4]
        T_states = np.stack(mini_batch[:,0])
        T_actions = np.stack(mini_batch[:,1])
        T_rewards = np.stack(mini_batch[:,2])
        T_next_states = np.stack(mini_batch[:,3])
        T_dones = np.stack(mini_batch[:,4])
        
        T_states = torch.FloatTensor(T_states).to(self.device)
        T_actions = torch.LongTensor(T_actions).to(self.device)
        T_rewards = torch.FloatTensor(T_rewards).to(self.device)
        T_next_states = torch.FloatTensor(T_next_states).to(self.device)
        T_dones = torch.FloatTensor(T_dones).to(self.device)
        
        T_q = self.model(T_states)
        # T_q = torch.stack([T_q[i][T_actions[i]] for i in range(self.batch_size)])

        
        T_next_q = self.model(T_next_states)
        T_next_q = torch.max(T_next_q,-1)[0]
        TD_target = torch.zeros((self.batch_size, self.n_action)).to(self.device)
        
        for i in range(self.batch_size):
            TD_target[i][T_actions[i]] = T_rewards[i] + self.discount_factor*(1. - T_dones[i]) *T_next_q[i]

        TD_target = TD_target.detach()
        
        
        
        self.optimizer.zero_grad()        
        cost = self.loss(T_q, TD_target).mean()
        cost.backward()#Gradient calculation
        self.optimizer.step()#Gradient update
            
        
        
class MLP(nn.Module):
    def __init__(self,n_state, n_action):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_action)
        )       
        
    def forward(self, x):        
        return self.layers(x)
    


if __name__ =="__main__":
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Run on ", device)
    
    torch.manual_seed(777)
    if device == 'cuda':        
        torch.cuda.manual_seed_all(777)
    
    
    env = gym.make('CartPole-v1')
    
    # env._max_episode_steps = 10000
    
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    num_episode = 10000
    max_step = 200


    policy = DQN(n_state, n_action, device)
    score_list = []
    for i in range(num_episode):
        state = env.reset()
        state = np.reshape(state, [-1])        
        
        score = 0
        for t in range(max_step):  
            # env.render()
    
            action = policy.get_action(state,i)
            next_state, reward, done, _ = env.step(action)            
            
            reward = reward if not done else -100
            score += reward            
            
            policy.save_sample((state, action, reward, next_state, done))

            state = next_state
            
            if len(policy.replay_memory) >= policy.train_start:            
                policy.train()
                
            if done:
                score +=100
                break
            
        score_list.append(score)

        # if i > 10 and i % 10 ==1:
        #     for _ in range(policy.batch_size):
        #         policy.train()
        print("Episode: {} score: {}".format(i+1,score))
        
        if np.mean(score_list[-10:]) >= max_step:
            break
        
        
                
    


    env.close()