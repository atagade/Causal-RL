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

alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
gammas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

max_rewards_qlearn = 0
max_rewards_causal_qlearn = 0
a_max = 0
g_max = 0

def qlearn(alpha, gamma):
	
	n_episodes = 1000
	n_steps = 100
	epsilon = 0.9
	max_epsilon = 1
	min_epsilon = 0.01
	decay_rate = 0.01
	flag = 0

	training_rewards = []

	for episode in range(n_episodes):
		state = env.reset()
		total_reward = 0
		
		for step in range(n_steps):
			if(epsilon > np.random.uniform(0,1)):
				action = env.action_space.sample()
			else:
				action = np.argmax(qtable[state,:])

			next_state, reward, done, info = env.step(action)
			#print(reward)
			total_reward += reward
			qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[next_state, :]) - qtable[state, action])
			state = next_state

			if(done):
				if(flag == 0 and total_reward >= 9):
					optimum_reward_ep = episode+1
					flag = 1
				break

		training_rewards.append(total_reward/step+1)
		epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
		
		#if(episode%10 == 0):
			#print('Average reward per episode: ' +str(sum(training_rewards[:-10])/10))

	#print('Optimum reward received at epoch: ' +str(optimum_reward_ep))

	#plt.plot([i for i in range(n_episodes)], training_rewards)
	#plt.show()

	return training_rewards

def causal_qlearn(alpha,gamma):

	env = gym.make('Taxi-v3')
	n_actions = env.action_space.n
	n_states = env.observation_space.n

	qtable = np.zeros((n_states,n_actions))

	n_episodes = 1000
	n_steps = 100
	epsilon = 0.9
	max_epsilon = 1
	min_epsilon = 0.01
	decay_rate = 0.01
	training_rewards = []
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
	 	
		training_rewards.append(total_reward/step+1)
		
		epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

	return training_rewards

for a in alphas:
	for i,g in enumerate(gammas):
		#print(i)
		rewards_qlearn = qlearn(a,g)
		rewards_causal_qlearn = causal_qlearn(a,g)

		if(max_rewards_qlearn < sum(rewards_qlearn)):
			max_rewards_qlearn = sum(rewards_qlearn)
			a_max_qlearn = a
			g_max_qlearn = g
		
		if(max_rewards_causal_qlearn < sum(rewards_causal_qlearn)):
			max_rewards_causal_qlearn = sum(rewards_causal_qlearn)
			a_max_causal_qlearn = a
			g_max_causal_qlearn = g

print('Maximum reward for qlearn is obtained when alpha = ' + str(a_max_qlearn) + ' and gamma = ' + str(g_max_qlearn))
print('Maximum reward for causal qlearn is obtained when alpha = ' + str(a_max_causal_qlearn) + ' and gamma = ' + str(g_max_causal_qlearn))




	
