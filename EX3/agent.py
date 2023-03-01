import numpy as np
import random
import matplotlib.pyplot as plt

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

	def play_episode(self, explore = True, learn = True):
		
		sequence = []
		self.env.start()

		# Play an episode
		while not self.env.done:
			
			state = self.env.state
			
			if explore:
				# Get action based on state and either policy or random based on epsilon
				action = self.explore(self.policy[state])
			else:
				action = self.policy[state]

			# Get step reward
			reward, new_state, done = self.env.step(action)

			# Add new state-action pair to sequence
			sequence.append((state, action, reward))

		if learn:
			# Get returns for state-action pairs
			returns = np.zeros(len(sequence))
			for i in reversed(range(len(sequence))):
				for j in range(i + 1):
					returns[j] += sequence[i][2]

		# Get new action values for state-action pairs 
		for seq, ret in zip(sequence, returns):
			state_action = seq[0] + (seq[1],)
			# (Return - prev_action_value) / (action_count + 1)
			self.action_values[state_action] = (ret - self.action_values[state_action]) / (self.action_counts[state_action] + 1)
			self.action_counts[state_action] += 1
		
		return returns[0], sequence

	# Update policys for state-action pairs
	def update_policy(self):
		self.policy = np.argmax(self.action_values, axis=-1)

	# Pick an action on random or based on policy
	def explore(self, action):
		if np.random.uniform(0, 1) < self.epsilon:
			return random.choice(self.acc_actions)
		else:
			return self.acc_actions[action]

	# Reset the learning
	def reset(self):
		self.action_values = \
		np.zeros((self.env.track.shape[1], self.env.track.shape[0], NUM_SPEEDS, NUM_SPEEDS,
					NUM_ACTIONS), dtype=np.float32)
		self.action_counts = \
		np.zeros((self.env.track.shape[1], self.env.track.shape[0], NUM_SPEEDS, NUM_SPEEDS,
					NUM_ACTIONS), dtype=np.int32)
		self.policy = np.full((self.env.track.shape[1], self.env.track.shape[0], NUM_SPEEDS, NUM_SPEEDS), 3,
							dtype=np.int32)
	
	def show_sequence(self, sequence, save_path=None, show_legend=True):
		track = self.env.track.copy()

		for item in sequence:
			state = item[0]
			track[state[1], state[0]] = 4

		im = plt.imshow(track)

		plt.axis("off")

		if save_path is not None:
			plt.savefig(save_path, bbox_inches="tight")

		plt.show()
