import numpy as np

class Agent:

	NO_ACTIONS = 9
	ACTIONS_TO_ACCELERATION = np.array([[1, 1], [0, 1], [1, 0], [0, 0], [-1, 0], [0, -1], [1, -1], [-1, 1], [-1, -1]])

	def __init__(self) -> None:
		pass

	def possible_new_velocity_states(self, velocity):
		pass

	