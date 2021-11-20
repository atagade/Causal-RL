import numpy as np
import networkx as nx
import random
import timeit
import gym
import matplotlib.pyplot as plt
from utils import get_node_values
from utils import interventional_selection

env = gym.make('Taxi-v3')
n_actions = env.action_space.n
n_states = env.observation_space.n

qtable = np.zeros((n_states,n_actions))

n_episodes = 1000
n_steps = 100
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.01
alpha = 0.8
gamma = 0.9
average_training_rewards = []
flag = 0
steps = []
G = nx.read_graphml('taxi_graph.graphml')


for episode in range(n_episodes):
	state = env.reset()
	total_reward = 0
	for step in range(n_steps):
		flag_a = 1
		observables = [i for i in env.decode(state)]
		V = get_node_values(observables)
		action = interventional_selection(state, G, V)
		#print(action)	
		if(action == None):
			flag_a = 0
			if(epsilon > np.random.uniform(0,1)):
				while(action == None or action >= 4):
					action = env.action_space.sample()
				#print(action)
			else:
				action = np.argmax(qtable[state,:])
				#print(action)
		next_state, reward, done, info = env.step(action)
		#print(reward)		
		total_reward += reward
		qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[next_state, :]) - qtable[state, action])
		state = next_state

		if(done):
			if(flag == 0 and total_reward >= 9):
				optimum_reward_ep = episode+1
				flag = 1
			#print('Episode_'+str(episode+1)+' Total reward: '+str(total_reward))
			break
 	
	average_training_rewards.append(total_reward/step+1)
	steps.append(step)
	
	if(flag_a == 0):
		epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
	
	#if(episode%10 == 0):
	#	print('Average reward at episode ' + str(episode+1) + ' : ' +str(sum(average_training_rewards[:-10])/10))

print('Optimum reward achieved at episode: ' +str(optimum_reward_ep))

plt.ylim(-5,5)
plt.xlabel('Episode number')
plt.ylabel('Total reward')
plt.plot(average_training_rewards)
#plt.plot([i for i in range(n_episodes)], steps)

plt.savefig('observations/taxi_causal_greedy_axisfixed.png')
