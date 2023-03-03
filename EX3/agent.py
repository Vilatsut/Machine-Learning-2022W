import numpy as np
import random
import matplotlib.pyplot as plt

NUM_SPEEDS = 5
NUM_ACTIONS = 9

class Agent:
	def __init__(self, env, epsilon, y) -> None:
		
		self.env = env
		self.epsilon = epsilon
		self.y = y
		self.Q_vals = np.random.rand(env.track.shape[1],env.track.shape[0],5,5,9)*400 - 500
		self.C_vals = np.zeros((env.track.shape[1],env.track.shape[0],5,5,9))

		self.action_values = None
		self.action_counts = None
		self.policy = None
		self.reset()

		self.acc_actions = [(1, 1), (1, 0), (0, 1), (0, 0), (-1, 0), (0, -1), (1, -1), (-1, 1), (-1, -1)]

	def play_episode(self, start_state = None, explore = True, learn = True):
		
		sequence = []
		self.env.start()

		if start_state:
			self.env.state = start_state
			
		# Play an episode
		while not self.env.done:
			
			state = self.env.state
			
			if explore:
				# Get action based on state and either policy or random based on epsilon
				action = self.explore(self.policy[state])
			else:
				action = self.acc_actions[self.policy[state]]
				print(action)

			# Get step reward
			reward, new_state, done = self.env.step(action)

			# Add new state-action pair to sequence
			sequence.append((state, action, reward))

		returns = list(range(-1, -len(sequence), -1))
		if learn:
			G = 0
			W = 1
			T = len(sequence)-1
			for t in range(T-1,-1,-1):
				G = self.y * G + returns[t-1]
				S_t = sequence[t][0]
				A_t = sequence[t][1]
				
				S_list = list(S_t)
				S_list.append(A_t)
				SA = tuple(S_list)
				
				self.C_vals[SA] += W
				self.Q_vals[SA] += (W*(G-self.Q_vals[SA]))/(self.C_vals[SA])           
				self.policy[S_t] = np.argmax(self.Q_vals[S_t])

		return returns[0], sequence

	# Update policys for state-action pairs
	def update_policy(self):
		pass
		# self.policy = np.argmax(self.action_values, axis=-1)

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
