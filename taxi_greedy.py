import numpy as np
import random
import timeit
import gym
import matplotlib.pyplot as plt

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
alpha = 0.8
gamma = 0.9
average_training_rewards = []
steps = []

flag = 0

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

	average_training_rewards.append(total_reward/(step+1))
	steps.append(step)
	epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
	
	#if(episode%10 == 0):
	#	print('Average reward at episode ' + str(episode+1) + ' : ' +str(sum(average_training_rewards[:-10])/10))
		#print('Average timestep at episode ' + str(episode+1) + ' : ' +str(sum(steps[:-10])/10))

print('Optimum reward achieved at episode: ' +str(optimum_reward_ep))

plt.ylim(-5,5)
plt.xlabel('Episode number')
plt.ylabel('Total reward')
plt.plot([i for i in range(n_episodes)], average_training_rewards)
#plt.plot([i for i in range(n_episodes)], steps)

plt.savefig('observations/taxi_greedy_axisfixed.png')
