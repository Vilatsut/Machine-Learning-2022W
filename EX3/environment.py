# State is (row-index, col-index, velocity-x, velocity-y) tuple
# Action is (change-x, change-y) tuple
import numpy as np
import random

TRACK = 0
WALL = 1
START = 2
FINISH = 3 

NOISE = True

class Environment:

    def __init__(self, racetrack: np.array) -> None:
        self.prev_state = (0,0,0,0)
        self.state = (0,0,0,0)
        self.velocity = (0,0)
        self.track: np.array = racetrack

        finish_y_coordinates, finish_x_coordinates = np.where(racetrack == 3)
        start_y_coordinates, start_x_coordinates = np.where(racetrack == 2)
        wall_y_coordinates, wall_x_coordinates = np.where(racetrack == 1)
        self.map = {
            "finish": list(zip(finish_x_coordinates, finish_y_coordinates)),
            "start": list(zip(start_x_coordinates, start_y_coordinates)),
            "wall": list(zip(wall_x_coordinates, wall_y_coordinates)),
        }
        self.episode = {
            "actions": np.array([]),
            "states": np.array([]),
            "rewards": np.array([]),
            "propabilities": np.array([])
        }
        self.step_count = 0

    def __get_new_state(self, action: tuple[int, int]):
        
        self.prev_state = self.state
        if not NOISE:
            velocity = self.__correct_velocity((self.state[2] + action[0], self.state[3] + action[1]))
        elif random.random() < 0.9:
            velocity = self.__correct_velocity((self.state[2] + action[0], self.state[3] + action[1]))
        else:
            velocity = (self.state[2], self.state[3])

        self.state = (
            self.state[0] + self.state[2],
            self.state[1] - self.state[3],
            velocity[0],
            velocity[1]
        )
        
    def __correct_velocity(self, velocity):
        
        # Make sure velovcity is within bounds
        if velocity[0] < 0:
            velocity = (0, velocity[1])
        elif velocity[0] > 4:
            velocity = (4, velocity[1])

        if velocity[1] < 0:
            velocity = (velocity[0], 0)
        elif velocity[1] > 4:
            velocity = (velocity[0], 4)

        # Make sure the velocity is not (0, 0)
        if velocity == (0, 0):
            if random.choice([True, False]):
                velocity = (1, 0)
            else:
                velocity = (0, 1)

        return velocity

    def __is_finished(self) -> bool:

        old_coord = self.prev_state[0:2]
        new_coord = self.state[0:2]

        walked_rows = np.array(range(old_coord[0], new_coord[0] + 1))
        walked_cols = np.array(range(new_coord[1], old_coord[1] + 1))
        fin = [x for x in self.map["finish"]]
        row_col_matrix = [(x,y) for x in walked_rows for y in walked_cols]
        intersect = [x for x in row_col_matrix if x in fin]

        return len(intersect) > 0

    def __is_out_of_bounds(self) -> bool:
        
        coord = (self.state[0:2])
        if coord[0] < 0 or coord[0] >= self.track.shape[1] or coord[1] < 0 or coord[1] >= self.track.shape[0]:
            print(coord[0])
            print(coord[1])
            print(self.track.shape[0])
            print(self.track.shape[1])
            return True
        else:
            return self.track[coord[1]][coord[0]] == 1

    # Reset the episode
    def reset(self):
        self.episode = {
            "actions": np.array([]),
            "states": np.array([]),
            "rewards": np.array([]),
            "propabilities": np.array([])
        }
        self.step_count = 0

    # Reset the car position and velocity
    def __start(self):
        random_start_coord = random.choice(self.map["start"])
        self.state = (
            random_start_coord[0],
            random_start_coord[1],
            0,
            0
        )

    def step(self, action):
        
        np.append(self.episode["actions"], action)
        reward = -1

        self.__get_new_state(action)

        if self.__is_finished():
            np.append(self.episode["rewards"], reward)
            np.append(self.episode["states"], self.state)
            self.step_count += 1
            return None, self.state
        
        elif self.__is_out_of_bounds():
            self.__start()

        np.append(self.episode["rewards"], reward)
        np.append(self.episode["states"], self.state)
        self.step_count += 1

        return reward, self.state
