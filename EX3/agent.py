import numpy as np


class MonteCarlo:

	acc_actions = np.array([[1, 1], [0, 1], [1, 0], [0, 0], [-1, 0], [0, -1], [1, -1], [-1, 1], [-1, -1]])
	no_acc_actions = 9
	no_speeds = 5

	def __init__(self, env, epsilon) -> None:
		
		self.env = env
		self.epsilon = epsilon

		self.action_values = None
		self.action_counts = None
		self.policy = None
		self.reset()

	def run_episode(self):
		
		episode = []

		while not self.env.is_finished():
			pass


	def reset(self):
		self.action_values = np.zeros((
			self.env.track.shape[0],
			self.env.track.shape[1],
			self.no_speeds,
			self.no_speeds,
			self.no_acc_actions
			), dtype=np.float32)
		self.action_counts = np.zeros((
			self.env.track.shape[0],
			self.env.track.shape[1],
			self.no_speeds,
			self.no_speeds,
			self.no_acc_actions			
		), dtype=np.int32)
		self.policy = np.zeros((
			self.env.track.shape[0],
			self.env.track.shape[1],
			self.no_speeds,
			self.no_speeds
		), dtype=np.int32)

	def update_policy(self):
		self.policy = np.argmax(self.action_values, axis=-1)

	def explore(self, action):
		
		if np.random.uniform(0, 1) < self.epsilon:
			return np.random.choice(self.acc_actions)
		else:
			return action
