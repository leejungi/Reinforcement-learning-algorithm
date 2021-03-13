# Cartpole
Cartpole-v0: max step=200  
Cartpole-v1: max step=200  
To increase max step, use 'env._max_episode_steps = 1000'

# 13DQN
Q(s,a) = DQN(s)  
Q(n_s,n_a) = DQN(n_s)
loss = (Q(s,a) - TD_target)^2  
TD_target = Reward + discount_factor*argmax(Q(n_s,n_a))

Experience when i make cartpole DQN code
1. Done reward = -100 is important when i use reward = -1, it is hard to train.
2. Becareful about loss(Crossentropy doesn't work) -> debugging label shape make trouble
3. Torch Tensor is important. Torch DNN input and model should be same device.
4. When sampling state from replay memory, output of slicing using numpy is object not floating array. So this can't be converted into torch.Tensor. Use np.stack to make object to floating array.

# 15DQN
Q(s,a) = DQN(s)  
Q'(n_s,n_a) = Target_DQN(n_s)  
loss = (Q(s,a) - TD_target)^2  
TD_target = Reward + discount_factor*argmax(Q'(n_s,n_a))

Experience
1. Replay memory size is important 
2. it may takes long time to learn cartpole but sometimes it takes 250 step for 10000 max epsiode step.
3. when hidden layer size was 32 or 64, it is hard to learn. But which 3 hidden layer makes easy to learn.
4. Exploration is important. Depending on exploration, DQN can learn within 250~3000 episode.

# DDQN
Update 2021-02-07: Prediction has problem(Q(n_s,n_a) is used not Q(s,a))
Q(s,a) = DQN(s)  
Q'(n_s,n_a) = Target_DQN(n_s)  
loss = (Q(s,a) - TD_target)^2  
TD_target = Reward + discount_factor*Q'(n_s, argmax(Q(n_s,n_a)))

Pick action in model(Q) by argmax.  
This action index is used in target model Q value to pick action.

Experience
1. DDQN works with changing TD_target in 15DQN. 
2. But i don't know DDQN is better than DQN. Because it shows similar performance comparing to 15DQN. This may be affected by exploration. Because the result is different in every time. ->  This may be caused by wrong prediction.

# A2C
Update 2021-03-09: Policy-based RL

Experience  
1. It is important to normalize returns and advantage function.

# PPO
Update 2021-03-10: 

Experience
1. r + gamma * V(st+1) can be converted into Rt.  
2. Some people doesn't use GAE.  
3. A2C needs normalizing reward and advantage but PPO may not need.  
4. next_states has same information in states. so this variable should be removed
5. With 1 step TD, PPO can be reached at 10,000 steps. But with GAE, it can't.

# Result
Comparing 13 DQN vs 15 DQN  
15 DQN can learn 10000 step environment.   
13 DQN is hard to learn 10000 but can learn 500 environment. 


