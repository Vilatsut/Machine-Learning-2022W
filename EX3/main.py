from racetracks import RACETRACK_1, RACETRACK_2
from environment import Environment
from agent import Agent

NO_EPISODES = 5
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

        # Evaluate after every EVALUATION_FREQ episode
        returns = []
        if ep_idx > 0 and ep_idx % EVALUATION_FREQ == 0:
            pass
            # Change episodes to be played without random actions and learning 
            # -> take returns and print average and print them out 
            #TODO
    # #TODO Print optimal routes from every starting position

ret = agent.play_episode(explore=False, learn=False)
agent.show_sequence(ret[1])
