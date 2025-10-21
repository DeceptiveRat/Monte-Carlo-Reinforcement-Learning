#!/bin/python3 

import sys
import getopt
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def usage():
	print("usage:", sys.argv[0])
	print("options:")
	print("-h: display this help screen")
	print("-r <int>: test environment row size (default = 5)")
	print("-c <int>: test environment column size (default = 5)")
	print("-e <float>: set epsilon (default = 0.3)")
	print("-n <int>: set number of episodes (default = 10000)")

def random_action(actions, state, row_size, column_size):
	while(True):
		# choose random action
		action = random.choice(actions)

		# make sure action is possible
		if action == "up":
			if state[0] > 0:
				return 0
		elif action == "down":
			if state[0] < row_size -1:
				return 1
		elif action == "left":
			if state[1] > 0:
				return 2
		elif action == "right":
			if state[1] < column_size - 1:
				return 3
			
def perform_action(action, state, rewards):
	new_state = state.copy()
	if action == 0:
		new_state[0]-=1
	elif action == 1:
		new_state[0]+=1
	elif action == 2:
		new_state[1]-=1
	elif action == 3:
		new_state[1]+=1
	
	reward = got_reward(new_state, rewards)
	if reward != 0:
		return new_state, reward, True
	else:
		return new_state, 0, False
	
def got_reward(state, rewards):
	for reward in rewards:
		if (state[0],state[1]) == reward[0]:
			return reward[1]
	
	return 0

def visualize_path(results_list, row_size, column_size):
	grid = np.zeros((row_size, column_size))
	for state, action in results_list:
		grid[state[0]][state[1]] = action + 1

	print("="*100)
	print(len(results_list))
	for r in range(row_size):
		for c in range(column_size):
			if grid[r][c] == 0:
				print(". ", end = "")
			elif grid[r][c] == 1:
				print("↑ ", end = "")
			elif grid[r][c] == 2:
				print("↓ ", end = "")
			elif grid[r][c] == 3:
				print("← ", end = "")
			elif grid[r][c] == 4:
				print("→ ", end = "")
		print("")
	print("="*100)

try:
	opts, args = getopt.getopt(sys.argv[1:], "hr:c:e:n:")
except getopt.GetoptError as err:
	print(err)
	usage()
	sys.exit(2)

actions = ["up", "down", "right", "left"]
# parameters ============
row_size = 5
column_size = 5
epsilon = 0.3
num_episodes = 10000
rewards = [((1,1), 3), ((0,3), 70), ((4,4),100000), ((2, 3), 5000)]
# ======================
avg_reward_per_action = [0] *5
episode_count = [0]*5

for option, argument in opts:
	if option == "-h":
		usage()
		sys.exit()
	elif option == "-r":
		row_size = int(argument)
	elif option == "-c":
		column_size = int(argument)
	elif option == "-e":
		epsilon = float(argument)
	elif option == "-n":
		num_episodes = int(argument)
	else:
		assert False, "unhandled option"

Q = np.zeros((row_size,column_size, len(actions)))
visit_count = np.zeros((row_size, column_size, len(actions)))

for episode in range(num_episodes):
	# reset variables
	state = [0, 0]
	result = 0
	results_list = []
	results_sum = 0
	reward = 0
	done = False
	count = 0

	# complete episode
	while not done:
		if np.random.rand() > epsilon:
			action = np.argmax(Q[state[0], state[1], :])
			# if max is 0, choose randomly
			if Q[state[0], state[1], action] == 0:
				action = random_action(actions, state, row_size, column_size)
		else:
			action = random_action(actions, state, row_size, column_size)
		new_state, reward, done = perform_action(action, state, rewards)
		results_list.append((state, action))
		state = new_state.copy()
		results_sum += reward
		count+=1
	
	# save values
	index = math.floor(math.log10(episode+1))
	episode_count[index]+=1
	avg_reward_per_action[index] += 1/episode_count[index]*(reward/count - avg_reward_per_action[index])
	#print(f"{episode}: {reward/count}")

	# update Q matrix
	for (state, action) in results_list:
		visit_count[state[0], state[1], action] += 1
		alpha = 1/visit_count[state[0], state[1],action]
		Q[state[0], state[1], action] += alpha*(results_sum - Q[state[0], state[1], action])
	
	# print path
	#visualize_path(results_list, row_size, column_size)

# print results
for i in range(len(avg_reward_per_action)):
	print(f"range: ~10^{i+1} - average reward per action: %.3f data size: %d" %(avg_reward_per_action[i], episode_count[i]))

# visualize results 
plt.bar(["1~9", "10~99", "100~9999", "10000~99999", "100000~999999"], avg_reward_per_action)
plt.show()
