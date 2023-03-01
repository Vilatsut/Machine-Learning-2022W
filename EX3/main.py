from racetracks import RACETRACK_1, RACETRACK_2
from environment import Environment
from agent import Agent

NO_EPISODES = 20
EVALUATION_FREQ = 1

if __name__ == "__main__":
    env = Environment(RACETRACK_1)
    env.show_racetrack()
    epsilon = 0.9
    agent = Agent(env, epsilon)

    for ep_idx in range(NO_EPISODES):
        print(f'Running episode number {ep_idx+1}...')
        ret = agent.play_episode()
        agent.update_policy()
        #agent.show_sequence(ret[1])
        print(f'Episode number {ep_idx+1} done.')

    print("Playing episode without learning")
    ret = agent.play_episode(explore=False, learn=False)
    print("Episode played.")
    agent.show_sequence(ret[1])
