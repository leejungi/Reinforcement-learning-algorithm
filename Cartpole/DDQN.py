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
class DDQN:
    def __init__(self,n_state, n_action, device='cpu'):
        #params
        self.n_state = n_state
        self.n_action = n_action
        self.device=device
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.discount_factor = 0.99
        self.learning_rate=0.001 #0.001
        self.batch_size= 64
        
        self.num_step =0
        self.num_exploration = 0
        self.num_train = 0
        self.model_update_interval = 10
        
        self.train_start = 1000
        self.replay_memory_size = 10000
        self.replay_memory = deque(maxlen=self.replay_memory_size)        
        
        #model define
        #action-value function
        self.model = MLP(self.n_state, self.n_action).to(self.device)        
        self.loss = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        #target action-value function
        self.target_model = MLP(self.n_state, self.n_action).to(self.device)  
        
    def reset_params(self):
        self.num_step =0
        self.num_exploration = 0
        
    def save_model(self, path):
        torch.save(self.model.state_dict(),path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(save_path))
        
    def save_sample(self, sample):
        #sample = [state, action, reward, next_state, done]
        self.replay_memory.append(sample)


        
	#E-greedy in state
    def get_action(self, state, test= False):

        self.num_step +=1
        
        if test:
            epsilon = -1
        else:
            epsilon = self.epsilon
            
        if np.random.rand(1) < epsilon:    
            self.num_exploration+=1
            #action = random.randrange(self.n_action)
            action = env.action_space.sample()
        else:
            # T means torch.Tensor
            T_state = torch.FloatTensor(state).to(self.device)
            T_q = self.model(T_state) 
            action = np.argmax(T_q.to('cpu').detach().numpy())        
            
        return action
    
    def epsilon_update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
    #Update
    def train(self):

        self.num_train += 1
        
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
        # TD_target = self.model(T_states)

        
        #_ shows max value, T_next_q shows index of max value
        _, T_next_q = self.model(T_next_states).detach().max(1)
       
        T_next_tq = self.target_model(T_next_states).detach()
        T_next_tq = T_next_tq.gather(1, T_next_q.unsqueeze(1))
        T_next_a = T_next_tq.squeeze()

        TD_target = torch.zeros((self.batch_size, self.n_action)).to(self.device)
        
        
        for i in range(self.batch_size):
            TD_target[i][T_actions[i]] = T_rewards[i] + self.discount_factor*(1. - T_dones[i]) *T_next_a[i]

        TD_target = TD_target.detach()       
        
        self.optimizer.zero_grad()        
        cost = self.loss(T_q, TD_target).mean()
        cost.backward()#Gradient calculation
        self.optimizer.step()#Gradient update
        
        if self.num_train%self.model_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        
class MLP(nn.Module):
    def __init__(self,n_state, n_action):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_state, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            nn.Linear(24, n_action)
        )       
        
    def forward(self, x):        
        return self.layers(x)
    
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
    step_list = []

    #Training        
    for i in range(num_episode):
        
        state = env.reset()
        state = np.reshape(state, [-1])    
        
        policy.reset_params()

        step = 0
        for t in range(max_step):  
            # env.render()
            

            action = policy.get_action(state)
            next_state, reward, done, _ = env.step(action)            
            
            if done and t != max_step -1:
                reward = -100
            
            step += 1
            
            policy.save_sample((state, action, reward, next_state, done))

            state = next_state
            
            if len(policy.replay_memory) >= policy.batch_size:  
                policy.epsilon_update()
                policy.train()
                
            if done:
                break
            
        step_list.append(step)


        print("Episode: {} step: {} #exploration: {:0.3f} epsilon: {:0.3f}".format(i+1,step, policy.num_exploration/policy.num_step,policy.epsilon))
        
        if (i+1)%100 == 0:
            step = test(policy, env)
            print("Test step: ",step)
            
        if np.mean(step_list[-10:]) >= max_step:
            break
        
    policy.save_model(save_path)
    
    plt.plot([i for i in range(len(step_list))],step_list)
    plt.show()
    
    
    # Test
    step = test(policy, save_path, env, True)
    print("Test step: ", step)

    env.close()