# Cartpole
Cartpole-v0: max step=200  
Cartpole-v1: max step=200  
To increase max step, use 'env._max_episode_steps = 1000'

# 13DQN
Q(s,a) = DQN(s)  
loss = (Q(s,a) - TD_target)^2  
TD_target = Reward + discount_factor*argmax(Q(n_s,n_a))

Experience when i make cartpole DQN code
1. Done reward = -100 is important when i use reward = -1, it is hard to train.
2. Becareful about loss(Crossentropy doesn't work)
3. Torch Tensor is important. Torch DNN input and model should be same device.
4. When sampling state from replay memory, output of slicing using numpy is object not floating array. So this can't be converted into torch.Tensor. Use np.stack to make object to floating array.

# 15DQN
# DDQN
