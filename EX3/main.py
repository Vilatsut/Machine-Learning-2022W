from racetracks import RACETRACK_1, RACETRACK_2
from environment import Environment
from agent import Agent

NO_EPISODES = 60000
EVALUATION_FREQ = 1

if __name__ == "__main__":
    env = Environment(RACETRACK_1)
    env.show_racetrack()
    epsilon = 0.1
    y = 1
    agent = Agent(env, epsilon, y)

    for ep_idx in range(NO_EPISODES):
        print(f'Running episode number {ep_idx+1}...')
        ret = agent.play_episode()
        agent.update_policy()
        print(f'Episode number {ep_idx+1} done.')

    for start in env.map["start"]:
        print(f"Playing from start coordinate: {start}")
        start_state = (*start, 0,0) 
        ret = agent.play_episode(start_state=start_state, explore=False, learn=False)
        print(f"Episode played with returns {ret[0]} and sequence {ret[1]}")
        agent.show_sequence(ret[1])