
class Environment:

    def __init__(self) -> None:
        pass

    def get_new_state(self, state, action):
        pass

    def is_finnished(self, state, action) -> bool:
        pass

    def is_out(self, state, action) -> bool:
        pass

    # Reset the episode
    def reset(self):
        pass

    # Reset the car position
    def start(self):
        pass

    def step(self, state, action):
        pass

    