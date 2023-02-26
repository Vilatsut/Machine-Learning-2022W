from racetracks import RACETRACK_1, RACETRACK_2
from environment import Environment
from agent import Agent

NO_EPISODES = 1
EVALUATION_FREQ = 1

if __name__ == "__main__":
    env = Environment(RACETRACK_1)
    epsilon = 0.9
    agent = Agent(env, epsilon)

    for episode_idx in range(NO_EPISODES):
        agent.play_episode()
        agent.update.policy()

        
        # Evaluate every EVALUATION_FREQ episode
        if episode_idx > 0 and episode_idx % EVALUATION_FREQ == 0:
            pass #TODO

    

