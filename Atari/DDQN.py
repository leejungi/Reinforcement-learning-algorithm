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
from skimage.color import rgb2gray
from skimage.transform import resize
from utils_memory import ReplayMemory
import torch.nn.functional as F

class DDQN:
    def __init__(self, n_state, n_action, device='cpu'):
        #params
        self.n_state = n_state
        self.n_action = n_action
        self.device=device
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.discount_factor = 0.99
        self.learning_rate= 0.000625 #0.001
        self.batch_size= 32
        
        self.num_step =0
        self.num_exploration = 0
        self.num_train = 0
        self.model_update_interval = 10000
        
        self.train_start = 50000
        self.replay_memory_size = 100000
        # self.replay_memory = deque(maxlen=self.replay_memory_size)        
        
        self.memory = ReplayMemory(4 + 1, self.replay_memory_size, self.device)
        
        #model define
        #action-value function
        self.model = CNN(self.n_state, self.n_action).to(self.device)        
        self.loss = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1.5e-4)
           
        #target action-value function
        self.target_model = CNN(self.n_state, self.n_action).to(self.device)  
        
        self.model.apply(CNN.init_weights)
        self.target_model.eval()
        
        self.rand = random.Random()
        self.rand.seed(0)
        
    def reset_params(self):
        self.num_step =0
        self.num_exploration = 0
        
    def save_model(self, path):
        torch.save(self.model.state_dict(),path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(save_path))
        
    # def save_sample(self, sample):
    def save_sample(self, state_queue, action, reward, done):
        #sample = [state, action, reward, next_state, done]
        self.memory.push(state_queue, action, reward, done)
        # self.replay_memory.append(sample)


        
	#E-greedy in state
    def get_action(self, state, test= False):

        self.num_step +=1
        
        if test:
            epsilon = self.epilson_min
        else:
            epsilon = self.epsilon
            
        if np.random.rand(1) < epsilon:    
            self.num_exploration+=1
            #action = random.randrange(self.n_action)
            # action = env.action_space.sample()
            action= self.rand.randint(0, self.n_action-1)
        else:
            # T means torh.Tensor
            # print(np.shape(list(state)[1:]))
            with torch.no_grad():
                # T_state = torch.FloatTensor([list(state)[1:]]).to(self.device)            
            # T_state = T_state.permute(0, 3, 1, 2)
                T_q = self.model(state) 
                action = np.argmax(T_q.to('cpu').detach().numpy())        
        
        # action += 1
        return action
    
    def epsilon_update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
    #Update
    def train(self):
        
        state_batch, action_batch, reward_batch, next_batch, done_batch = self.memory.sample(self.batch_size)

        values = self.model(state_batch.float()).gather(1, action_batch)
        values_next = self.target_model(next_batch.float()).max(1).values.detach()
        expected = (self.discount_factor * values_next.unsqueeze(1)) * (1. - done_batch) + reward_batch
        loss = F.smooth_l1_loss(values, expected)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.num_train += 1
        
        # mini_batch = random.sample(self.replay_memory, self.batch_size)
        
        # #print(np.shape(np.array(mini_batch[:,0])))

        # # print(np.shape(mini_batch))
        # mini_batch = np.array(mini_batch)
        

        # T_states = np.stack(mini_batch[:,0])[:,:4]
        # T_actions = np.stack(mini_batch[:,1])
        # T_rewards = np.stack(mini_batch[:,2])
        # T_next_states = np.stack(mini_batch[:,0])[:,1:]
        # T_dones = np.stack(mini_batch[:,3])
        
        # T_states = torch.FloatTensor(T_states).to(self.device)
        # T_actions = torch.LongTensor(T_actions).to(self.device)
        # T_rewards = torch.FloatTensor(T_rewards).to(self.device)
        # T_next_states = torch.FloatTensor(T_next_states).to(self.device)
        # T_dones = torch.FloatTensor(T_dones).to(self.device)
        
        
        # T_q = self.model(T_states)        

        # #_ shows max value, T_next_q shows index of max value
        # _, T_next_q = self.model(T_next_states).detach().max(1)
       
        # T_next_tq = self.target_model(T_next_states).detach()
        # T_next_tq = T_next_tq.gather(1, T_next_q.unsqueeze(1))
        # T_next_a = T_next_tq.squeeze()

        # TD_target = torch.zeros((self.batch_size, self.n_action)).to(self.device)
        
        
        # for i in range(self.batch_size):
        #     TD_target[i][T_actions[i]] = T_rewards[i] + self.discount_factor*(1. - T_dones[i]) *T_next_a[i]

        # TD_target = TD_target.detach()       
        
        # self.optimizer.zero_grad()        
        # cost = self.loss(T_q, TD_target).mean()
        # cost.backward()#Gradient calculation
        # self.optimizer.step()#Gradient update
        
        if self.num_train%self.model_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        
class CNN(nn.Module):
    def __init__(self,n_state, n_action):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_state, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=n_action)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

def test(policy, env, max_step, save_path=None,rendering=False):
    
    frame_size = 4
    state_size = 84
    
    if save_path != None:
        policy.load_model(save_path)
        
    state = env.reset()
    state_deque = deque(maxlen=frame_size+1)   
    
    for _ in range(10):
        observe, _,_,_ = env.step(1)
        
    state = preprocessing(state, state_size)
    
    for _ in range(frame_size+1):
        state_deque.append(state)


    score = 0
    start_life=5 
    dead = False
    
    with torch.no_grad():
        for t in range(max_step):
            if rendering == True:
                env.render()
            
            state = torch.cat(list(state_deque)[1:]).unsqueeze(0)
            action = policy.get_action(state, False)
            
            if dead:
                dead =False
                action = 0
                
            next_state, reward, done, info = env.step(action+1)
            
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']
            
            next_state = preprocessing(next_state, state_size)
            state_deque.append(next_state)
            
            # reward = np.clip(reward, -1., 1.)
            
            score += reward
            # print(action, step, done, reward)

            if done:            
                break
            
    print("Test score: {}".format(score))

def preprocessing(state_set, state_size):
    new_state = np.uint8(
        resize(rgb2gray(state_set), (state_size, state_size), mode='constant') * 255)
    new_state = np.float32(new_state / 255.0)
    new_state=torch.from_numpy(new_state).view(1, 84, 84)
    return new_state.to(device).float()
    
if __name__ =="__main__":
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Run on ", device)
    
    torch.manual_seed(777)
    if device == 'cuda':        
        torch.cuda.manual_seed_all(777)
    
    
    env = gym.make('BreakoutDeterministic-v4')
    
    save_path='./save_model/DDQN.pth'
    num_episode = 50000
    max_step=5000000
    
    Train = True
    Test = True
    frame_size = 4
    state_size = 84
    
    n_action = 3 #env.action_space.n

    policy = DDQN(frame_size, n_action, device)
    score_list = []
    
    state_deque = deque(maxlen=frame_size+1)  
    state_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    if Train == True:
        #Training        
        for i in range(num_episode):
            
            state = env.reset()
            
            state = preprocessing(state, state_size)
            
            
            for _ in range(10):
                observe, _,_,_ = env.step(1)
            
            for _ in range(frame_size+1):
                state_deque.append(state)
            
            
            
            policy.reset_params()
    
            score, start_life = 0, 5
            dead = False
            
            for t in range(max_step):  
                
                state = torch.cat(list(state_deque)[1:]).unsqueeze(0)
                action = policy.get_action(state)
                
                if dead:
                    dead =False
                    action = 0
                    
                total_reward = 0
                for s in range(4):
                    next_state, reward, done, info = env.step(action+1)
                    # next_state = preprocessing(next_state, state_size)
                    if s == 4 - 2: state_buffer[0] = next_state
                    if s == 4 - 1: state_buffer[1] = next_state
                    total_reward += reward
                    if done:
                        break
                    
                next_state = state_buffer.max(axis=0)
                next_state = preprocessing(next_state, state_size)
                

                

                
                # next_state, reward, done, info = env.step(action+1)   
                
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                
                # next_state = preprocessing(next_state, state_size)
                state_deque.append(next_state)
                
                
                
                # reward = np.clip(reward, -1., 1.)
                
                score += total_reward
                
                #Put replay memory
                
                # policy.save_sample((state_deque, action, reward, done))

                policy.save_sample(torch.cat(list(state_deque)).unsqueeze(0), action, reward, done)
    
                if len(policy.memory) >= policy.train_start and t % 4==0:  
                    policy.epsilon_update()
                    policy.train()
                    
                if done:
                    break
                
            score_list.append(score)
    
    
            print("Episode: {} score: {} epsilon: {:0.3f}".format(i+1, score, policy.epsilon))
            
            if (i+1)%1000 == 0:
                policy.save_model('./save_model/DDQN_Breakout_'+str(i+1)+'.pth')
                test(policy, env, max_step,rendering=False)

                
        policy.save_model(save_path)
        
        plt.plot([i for i in range(len(score_list))],score_list)
        plt.show()
    
    if Test == True:
        # Test
        # save_path = './save_model/DDQN_Breakout_500.pth'
        test(policy, env, max_step, save_path,  True)
                

    env.close()