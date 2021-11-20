import numpy as np
import random
import timeit
import gym
import matplotlib.pyplot as plt
import statistics as stat

average_steps = []
optimum_reward_eps = []
failures = []

for i in range(10):

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
	average_episode_steps = []

	steps = []
	failure = 0
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
			if(reward == -10):
				failure += 1
			total_reward += reward
			qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[next_state, :]) - qtable[state, action])
			state = next_state

			if(done):
				if(flag == 0 and total_reward >= 9):
					optimum_reward_ep = episode+1
					optimum_reward_eps.append(optimum_reward_ep)
					flag = 1
				break

		#average_training_rewards.append(total_reward/(step+1))
		steps.append(step+1)
		epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

	#average_reward.append(sum(average_training_rewards)/1000)
	average_steps.append(sum(steps)/1000)
	failures.append(failure)

print('Average optimum reward episode: ' +str(sum(optimum_reward_eps)/10) + ' +- ' +str(stat.stdev(optimum_reward_eps)))
print('Average timestep per episode: ' + str(sum(average_steps)/10) + ' +- ' +str(stat.stdev(average_steps)))
print('Number of failures: ' +str(sum(failures)/10) + ' +- ' + str(stat.stdev(failures)))

	
