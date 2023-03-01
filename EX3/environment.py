import numpy as np
import random
import matplotlib.pyplot as plt

TRACK = 0
WALL = 1
START = 2
FINISH = 3 

NOISE = True

class Environment:

    # State is (row-index, col-index, velocity-x, velocity-y) tuple
    # Action is (change-x, change-y) tuple
    def __init__(self, racetrack: np.array) -> None:
        self.prev_state = (0,0,0,0)
        self.state = (0,0,0,0)
        self.track = racetrack
        
        self.done = False

        finish_y_coordinates, finish_x_coordinates = np.where(racetrack == 3)
        start_y_coordinates, start_x_coordinates = np.where(racetrack == 2)
        wall_y_coordinates, wall_x_coordinates = np.where(racetrack == 1)
        self.map = {
            "finish": list(zip(finish_x_coordinates, finish_y_coordinates)),
            "start": list(zip(start_x_coordinates, start_y_coordinates)),
            "wall": list(zip(wall_x_coordinates, wall_y_coordinates)),
        }

    # Calculate a new state based on current state and action
    def __get_new_state(self, action: tuple[int, int]):
        
        self.prev_state = self.state

        # If noise, 90% of the time it does the action, 10% it makes no change to state
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
    
    # Check velocity so it adheres to the rules specified
    def __correct_velocity(self, velocity):
        
        # Make sure velocity is within bounds of the map
        if velocity[0] < 0:
            velocity = (0, velocity[1])
        elif velocity[0] > 4:
            velocity = (4, velocity[1])

        # Make sure velocity is between 0 and 4
        if velocity[1] < 0:
            velocity = (velocity[0], 0)
        elif velocity[1] > 4:
            velocity = (velocity[0], 4)

        # Make sure the velocity is not (0, 0), if it is, change x or y randomly to 1
        if velocity == (0, 0):
            if random.choice([True, False]):
                velocity = (1, 0)
            else:
                velocity = (0, 1)

        return velocity

    # Check if path crosses the finish line
    def __is_finished(self) -> bool:

        old_coord = self.prev_state[0:2]
        new_coord = self.state[0:2]

        path = self.__find_straight_path(old_coord, new_coord)

        intersect = [x for x in path if x in self.map["finish"]]

        return len(intersect) > 0

    # Check if path goes through wall or outside of bounds
    def __is_out_of_bounds(self) -> bool:
        
        old_coord = self.prev_state[0:2]
        new_coord = self.state[0:2]

        path = self.__find_straight_path(old_coord, new_coord)

        intersect = [coord for coord in path if coord in self.map["wall"]]

        # If out of bounds
        if new_coord[0] < 0 or new_coord[0] >= self.track.shape[1] or new_coord[1] < 0 or new_coord[1] >= self.track.shape[0]:
            return True
        # If path intersects wall
        elif len(intersect) > 0:
            return True
        # If new coord is on wall 
        else:
            return self.track[new_coord[1]][new_coord[0]] == 1

    def __find_straight_path(self, start_coord, end_coord):
        # Unpack the start and end coordinates
        x0, y0 = start_coord
        x1, y1 = end_coord
        
        # Calculate the absolute differences between the coordinates
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # Determine the direction of the movement
        if x0 < x1:
            x_step = 1
        else:
            x_step = -1
        
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1
        
        # Calculate the error and the initial values of x and y
        error = dx - dy
        x = x0
        y = y0
        
        # Create a list to store the cells on the path
        path = []
        
        # Walk along the line using Bresenham's algorithm
        while x != x1 or y != y1:
            path.append((x, y))
            
            error_2 = error * 2
            
            if error_2 > -dy:
                error -= dy
                x += x_step
            
            if error_2 < dx:
                error += dx
                y += y_step
        
        # Add the last cell to the path
        path.append((x, y))
        
        return path

    # Reset the car position and velocity
    def start(self):
        random_start_coord = random.choice(self.map["start"])
        self.state = (
            random_start_coord[0],
            random_start_coord[1],
            0,
            0
        )
        self.done = False

    # Make an action on state
    def step(self, action):
        
        reward = -1

        self.__get_new_state(action)

        if self.__is_finished():
            self.done = True
            return reward, self.prev_state, self.done
        
        elif self.__is_out_of_bounds():
            self.start()

        return reward, self.prev_state, self.done
    
    def show_racetrack(self, save_path=None):
        im =  plt.imshow(self.track)

        plt.axis("off")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()
