import numpy as np
import random

class Maze:
	def __init__(self, height=10, width=10):
		self.height = height
		self.width = width

		#Make maze(Bug: the start point can be closed by the wall)
		self.maze = np.zeros((self.height,self.width))
		self.wall = [[random.randrange(0,self.height), random.randrange(0,self.width)] for i in range(min(self.height,self.width))]
		self.start = [0,0]
		self.end = [self.height-1,self.width-1]
		self.state = self.start

		for i in range(min(self.height,self.width)):
			self.maze[self.wall[i][0]][self.wall[i][1]]= 1 # wall: 1

		self.maze[self.start[0]][self.start[1]] = 2 # start point
		self.maze[self.end[0]][self.end[0]] = 3 #Goal

	def get_state(self):
		return self.state

	def get_hw(self):
		return self.height, self.width

	def reset(self):
		self.state = self.start

	def print_maze(self):
		for h in range(self.height):
			print(self.maze[self.height-h-1])

	def step(self, action):
		x= [0,0,-1,1]
		y= [1,-1,0,0]

		next_state = [y[action], x[action]]
		next_state[0] += self.state[0]
		next_state[1] += self.state[1]

		if next_state[0] < 0 or next_state[1] <0 or next_state[0] >=self.height or next_state[1] >= self.width or self.maze[next_state[0]][next_state[1]] == 1:
			reward = -100
			done = True
		elif self.maze[next_state[0]][next_state[1]] == 3:
			reward = 100
			done = True
		else:
			reward = 0
			done = False

		self.state = next_state

		return next_state, reward, done


	def load(self,file_name):
		f = open(file_name,"r")

		lines = f.readlines()

		self.maze =[]
		self.wall = []

		for n, line in enumerate(reversed(lines)):
			w=[]
			for i in range(len(line)-1):
				if int(line[i]) == 2:
					self.start= [n,i]
				elif int(line[i]) == 3:
					self.end = [n,i]
				elif int(line[i]) == 1:
					self.wall.append([n,i])

				w.append(int(line[i]))

			self.maze.append(w)
		
		self.state = self.start
		self.height, self.width = np.shape(self.maze)
		
			

