# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from networkforall import Network
from utilities import soft_update, convert_to_tensor, transpose_to_tensor, transpose_list
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.1):
        super(MADDPG, self).__init__()

        # DDGAgent used only to train independent actors
        self.maddpg_agent = [DDPGAgent(24, 256, 128, 2),
                             DDPGAgent(24, 256, 128, 2)]

        # Shared critic trained for both agents
        # critic input = obs_full + actions = 48+2+2=52
        self.critic = Network(52, 256, 128, 1).to(device)
        self.target_critic = Network(52, 256, 128, 1).to(device)

        # initialize targets same as original networks
        hard_update(self.target_critic, self.critic)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=1.0e-3, weight_decay=0.0)

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    """"
    def target_act(self, obs_all_agents, noise=0.0):
        # get target network actions from all the agents in the MADDPG object 
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions
    """

    def target_act(self, obs_all_agents_list, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions_list = []
        for obs_all_agents in obs_all_agents_list:
            target_actions = []
            for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents):
                target_actions.append(ddpg_agent.target_act(obs, noise))
            target_actions_list.append(torch.stack(target_actions))
        return target_actions_list

    def act_on_list(self, obs_all_agents_list, agent_number):
        actions_list = []
        for obs_all_agents in obs_all_agents_list:
            actions = []
            for i in range(len(self.maddpg_agent)):
                if i == agent_number:
                    actions.append(self.maddpg_agent[i].actor(obs_all_agents[i]))
                else:
                    actions.append(self.maddpg_agent[i].actor(obs_all_agents[i]).detach())
            actions_list.append(torch.stack(actions))
        return actions_list

    @staticmethod
    def convert_samples_to_tensor(samples):
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for sample in samples:
            obs.append(torch.tensor(sample[0], dtype=torch.float))
            actions.append(torch.tensor(sample[1], dtype=torch.float))
            rewards.append(torch.tensor(sample[2], dtype=torch.float))
            next_obs.append(torch.tensor(sample[3], dtype=torch.float))
            dones.append(torch.tensor(sample[4], dtype=torch.float))
        return obs, actions, rewards, next_obs, dones

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        obs_full, actions, rewards, next_obs_full, dones = self.convert_samples_to_tensor(samples)
        # print('next_obs_full: {}'.format(next_obs_full))

        obs_full_s = torch.stack(obs_full)
        # print('obs_full stacked shape: {}'.format(obs_full_s.shape))
        next_obs_full_s = torch.stack(next_obs_full)
        # print('next_obs_full stacked: {}'.format(next_obs_full_s))
        # print('next_obs_full stacked shape: {}'.format(next_obs_full_s.shape))

        obs_full_c = torch.reshape(obs_full_s, (len(samples), -1))
        # print('obs_full concatenated shape: {}'.format(obs_full_c.shape))
        # next_obs_full_c = torch.cat(next_obs_full)
        next_obs_full_c = torch.reshape(next_obs_full_s, (len(samples), -1))
        # print('next_obs_full concatenated: {}'.format(next_obs_full_c))
        # print('next_obs_full concatenated shape: {}'.format(next_obs_full_c.shape))

        agent = self.maddpg_agent[agent_number]
        self.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        #target_actions = self.target_act(next_obs)
        target_actions = self.target_act(next_obs_full_s)
        target_actions_s = torch.stack(target_actions)
        # print('target_actions stacked shape: {}'.format(target_actions_s.shape))
        target_actions_c = torch.reshape(target_actions_s, (len(samples), -1))
        # print('target_actions concatenated shape: {}'.format(target_actions_c.shape))

        target_critic_input = torch.cat((next_obs_full_c,target_actions_c), dim=1).to(device)
        # print('target_critic_input shape: {}'.format(target_critic_input.shape))

        with torch.no_grad():
            q_next = self.target_critic(target_critic_input)
            # print('q_next shape: {}'.format(q_next.shape))

        rewards_s = torch.stack(rewards)
        # print('rewards_s stacked shape: {}'.format(rewards_s.shape))
        dones_s = torch.stack(dones)
        # print('dones_s stacked shape: {}'.format(dones_s.shape))
        y = rewards_s[:,agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - dones_s[:,agent_number].view(-1, 1))
        # print('y shape: {}'.format(y.shape))

        # action = torch.cat(action, dim=1)
        actions_s = torch.stack(actions)
        # print('actions stacked shape: {}'.format(actions_s.shape))
        actions_c = torch.reshape(actions_s, (len(samples), -1))
        # print('actions concatenated shape: {}'.format(actions_c.shape))
        # critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        critic_input = torch.cat((obs_full_c, actions_c), dim=1).to(device)
        # print('critic_input shape: {}'.format(critic_input.shape))
        q = self.critic(critic_input)
        # print('q shape: {}'.format(q.shape))

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        q_input = self.act_on_list(obs_full_s, agent_number)
        q_input_s = torch.stack(q_input)
        # print('q_input stacked shape: {}'.format(q_input_s.shape))

        # q_input = torch.cat(q_input, dim=1)
        q_input_c = torch.reshape(q_input_s, (len(samples), -1))
        # print('q_input concatenated shape: {}'.format(q_input_c.shape))
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full_c, q_input_c), dim=1)

        # get the policy gradient
        actor_loss = -self.critic(q_input2).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
            
            
            




