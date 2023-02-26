import numpy as np
import random

NUM_SPEEDS = 5
NUM_ACTIONS = 9

class Agent:

	

	def __init__(self, env, epsilon) -> None:
		
		self.env = env
		self.epsilon = epsilon

		self.action_values = None
		self.action_counts = None
		self.policy = None
		self.reset()

		self.acc_actions = [(1, 1), (0, 1), (1, 0), (0, 0), (-1, 0), (0, -1), (1, -1), (-1, 1), (-1, -1)]

	def play_episode(self):
		
		sequence = []
		self.env.start()

		while not self.env.done:
			
			state = self.env.state

			action = self.explore(self.policy[state])

			reward, state, done = self.env.step(action)

			sequence.append((state, action, reward))
		
		print(sequence)

	def update_policy():
		pass

	def explore(self, action):

		if np.random.uniform(0, 1) < self.epsilon:
			return random.choice(self.acc_actions)
		else:
			return action

	def reset(self):

		self.action_values = \
		np.zeros((self.env.track.shape[1], self.env.track.shape[0], NUM_SPEEDS, NUM_SPEEDS,
					NUM_ACTIONS), dtype=np.float32)
		self.action_counts = \
		np.zeros((self.env.track.shape[1], self.env.track.shape[0], NUM_SPEEDS, NUM_SPEEDS,
					NUM_ACTIONS), dtype=np.int32)
		self.policy = np.zeros((self.env.track.shape[1], self.env.track.shape[0], NUM_SPEEDS, NUM_SPEEDS),
							dtype=np.int32)
