from unityagents import UnityEnvironment
import numpy as np
from maddpg import MADDPG
from buffer import ReplayBuffer
import torch
from collections import deque
from utilities import transpose_list, transpose_to_tensor, convert_to_tensor
import matplotlib.pyplot as plt


def main():
    env = UnityEnvironment(file_name="Tennis_Windows_x86_64\Tennis.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    agent = MADDPG()

    def seeding(seed=1):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def maddpg(n_episodes=50000, max_t=1000, print_every=100, batchsize=128):
        seeding()
        buffer = ReplayBuffer(int(50000 * max_t))
        noise = 2
        noise_reduction = 0.9999
        scores_deque = deque(maxlen=print_every)
        scores = []
        for i_episode in range(1, n_episodes + 1):
            scores_agents = np.zeros(num_agents)
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            # print('states from env: {}'.format(states))
            while True:
                # agent chooses actions
                states_converted_to_tensor = convert_to_tensor(states)
                # print('states converted to tensor: {}'.format(states_converted_to_tensor))
                actions = agent.act(states_converted_to_tensor, noise=noise)
                # print('actions from agent: {}'.format(actions))
                noise *= noise_reduction
                actions_array = torch.stack(actions).detach().numpy()
                # print('actions array: {}'.format(actions_array))

                # environment takes action and returns new states and rewards
                env_info = env.step(actions_array)[brain_name]
                next_states = env_info.vector_observations
                # print('next states from env: {}'.format(next_states))
                rewards = env_info.rewards
                # print('rewards from env: {}'.format(rewards))
                dones = env_info.local_done
                # print('dones from env: {}'.format(dones))

                # store in shared replay buffer
                experience = (states, actions_array, rewards, next_states, dones)
                # print('experience: {}'.format(experience))
                buffer.push(experience)

                # update agent with experience sample
                if len(buffer) > batchsize:
                    for a_i in range(2):
                        samples = buffer.sample(batchsize)
                        # print('samples: {}'.format(samples))
                        agent.update(samples, a_i)
                    agent.update_targets()  # soft update the target network towards the actual networks

                # update episode score with agent rewards
                scores_agents += rewards
                states = next_states
                if np.any(dones):
                    break
            scores_deque.append(np.max(scores_agents))
            scores.append(np.max(scores_agents))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) >= 0.5 and i_episode >= 100:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_deque)))
                for i, maddpg_agent in zip(range(num_agents), agent.maddpg_agent):
                    torch.save(maddpg_agent.actor.state_dict(), 'checkpoint_actor_{}.pth'.format(i))
                torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
                break
        return scores


    scores = maddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    plt.show()

    env.close()


if __name__=='__main__':
    main()