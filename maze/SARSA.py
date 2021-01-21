from Environment import *
import random
import matplotlib.pyplot as plt
import math

class SARSA_policy:
	def __init__(self, height =10, width =10):
		self.num_action = 4
		self.height = height
		self.width = width
		self.q_table = np.zeros((self.height,self.width,self.num_action))
		self.learning_rate = 0.01
		self.discount_factor = 0.9

	#Need 'next_action' comparing with Q-learning
	def learn(self,state, action, reward, next_state, next_action):

		q = self.q_table[state[0]][state[1]][action]

		if next_state[0] < 0 or next_state[1] <0 or next_state[0] >=self.height or next_state[1] >= self.width:
			td_target = reward
		else:
			td_target = reward + self.discount_factor * self.q_table[next_state[0]][next_state[1]][next_action]

		self.q_table[state[0]][state[1]][action] += self.learning_rate * (td_target - q)

	def get_action(self, state, epsilon):
		if state[0] < 0 or state[1] <0 or state[0] >=self.height or state[1] >= self.width:
			return 0

		if np.random.rand() < epsilon:
			action = np.argmax(self.q_table[state[0]][state[1]])
		else:
			action = np.random.choice([0,1,2,3])

		return action



if __name__ =="__main__":

	maze = Maze(height=10, width=10)
	maze.load("maze.txt")
	height, width = maze.get_hw()
	policy = SARSA_policy(height, width)

	num_episode = 50000
	max_step = 10000
	epsilon = 0.7
	discount_factor = 0.9

	G_list =[0 for i in range(num_episode)]

	#Train
	for i in range(num_episode):
		maze.reset()

		for j in range(max_step):
			state = maze.get_state()

			action = policy.get_action(state, epsilon)

			next_state, reward, done = maze.step(action)

			next_action = policy.get_action(next_state, epsilon)

			policy.learn(state, action, reward, next_state, next_action)
			
			G_list[i] +=reward
#			G_list[i] += math.pow(discount_factor,j) * reward

			if done == True:
				break

	#Test
	maze.reset()
	Total_reward =0
	action_list=[]

	print("Test Start")
	for j in range(height*width):
		state = maze.get_state()

		action = policy.get_action(state,1)
		action_list.append(action)

		next_state, reward, done = maze.step(action)

		next_action = policy.get_action(next_state, 1)

		
		Total_reward +=reward

		if done == True:
			break

	maze.print_maze()
	print(action_list)
	print("Reward: ", Total_reward)

	plt.plot(G_list,'o')
	plt.show()
