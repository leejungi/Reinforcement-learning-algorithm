from Environment import *
import random
import matplotlib.pyplot as plt
import math

class MC_policy:
	def __init__(self, height =10, width =10):
		self.num_action = 4
		self.height = height
		self.width = width
		self.value_table = np.zeros((self.height,self.width))
		self.learning_rate = 0.01
		self.discount_factor = 0.9
		self.samples=[]

	def learn(self):

		Gt=0

		visited_state=[]
		#update using 1 episode
		for i,sample in enumerate(reversed(self.samples)):
			state, reward =sample
			if state not in visited_state:
				visited_state.append(state)

				Gt = reward + self.discount_factor*Gt
				y, x = state				
				if y < 0 or x <0 or y >=self.height or x >= self.width:
					value =0
				else:
					value= self.value_table[state[0]][state[1]]
					self.value_table[state[0]][state[1]] = value + self.learning_rate*(Gt-value)


	def save_sample(self, sample):
		self.samples.append(sample)

	def reset_sample(self):
		self.samples=[]		

	def get_action(self, state, epsilon):

		pos = [[1,0],[-1,0],[0,-1],[0,1]]
		value = [0,0,0,0]

		if np.random.rand() < epsilon:
			for i in range(len(value)):
				y=pos[i][0]+state[0]
				x=pos[i][1]+state[1]
				#if boundary, the value will be itself.
				if y < 0 or x <0 or y >=self.height or x >= self.width:
					value[i] = self.value_table[state[0]][state[1]]
				else:
					value[i] = self.value_table[y][x]

			action = np.argmax(value)
		else:
			action = np.random.choice([0,1,2,3])

		return action



if __name__ =="__main__":

	maze = Maze(height=5, width=5)
	maze.load("maze.txt")
	height, width = maze.get_hw()
	
	policy = MC_policy(height, width)

	num_episode = 50000
	max_step = 1000
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
			policy.save_sample([next_state, reward])

#			G_list[i] += math.pow(discount_factor,j) * reward
			G_list[i] += reward

			if done == True:
				break


		policy.learn()
		policy.reset_sample()
			
	#Test
	maze.reset()
	Total_reward =0
	action_list=[]

	print("Test Start")
	for j in range(height*width):
		state = maze.get_state()

		action = policy.get_action(state,1)
		action_list.append(action)

		_, reward, done = maze.step(action)
	
		Total_reward +=reward

		if done == True:
			break

	maze.print_maze()
	print(policy.value_table)
	print(action_list)
	print("Reward: ", Total_reward)

	plt.plot(G_list,'o')
	plt.show()
