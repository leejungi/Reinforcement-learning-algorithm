# SAC
Update 2021-03-14: 

Don't consider V network(it can be replaced by Q network)

1. I just apply action without scaling(the output action value of actor network has range from 0 to 1. so this value should be scaled)
2. I adopt replay buffer into 2 method. One is training all the replay buffer and the other is training partial replay buffer(with batch_size). But i think it is not important. Both of them need enough training epoch when training.

# PPO
Update 2021-03-26:

1. Reward normalizing is import. When i just use gym Pendulum reward, PPO can' make over -5.9 rewards(average). After normalizing, it works.

# Experience
1. PPO takes more time to work. I think this is because SAC use replay buffer. So if someone needs to be training agent in short, it would be better using SAC.  

# Reference
1. [PPO reference](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char07%20PPO/PPO_pendulum.py)
 -> this doesn't works, this code shows 5.8*200 average rewards. 
2. [PPO reference2](https://github.com/gouxiangchen/ac-ppo.git)
 -> this works, this need more time to be stable than SAC but works comparing to PPO reference 1.
3. [SAC reference](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch)
