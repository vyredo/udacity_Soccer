import torch
from QNetwork import QNetwork
from PrioritizedExperienceReplay import PrioritizedReplayBuffer
import numpy as np


class DistributedDQN:
    def __init__(self, DEVICE,
                 env,
                 brains, num_agents,
                 gamma, lr, tau,
                 buffer_size,
                 batch_size,
                 alpha, beta):

        self.brains = brains
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.device = torch.device(DEVICE)

        # Initialize agents per brain
        self.agents = {}
        for name, brain in brains.items():
            action_size = brain.vector_action_space_size
            state_size = brain.vector_observation_space_size
            num_agents = len(env.reset(train_mode=True)[name].agents)

            self.agents[name] = {
                'q_network': QNetwork(state_size, action_size).to(self.device),
                'target_q_network': QNetwork(state_size, action_size).to(
                    self.device),
                'optimizer': torch.optim.Adam(QNetwork(state_size,
                                                       action_size)
                                              .parameters(), lr=lr),
                'replay_buffer': PrioritizedReplayBuffer(
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    alpha=alpha),
                'num_agents': num_agents,
                'action_size': action_size,
                'state_size': state_size
            }

            # Copy weights to target networks
            self.agents[name]['target_q_network'].load_state_dict(
                self.agents[name]['q_network'].state_dict())

    def act(self, states, epsilon):
        actions = []
        for name, agent in self.agents.items():
            actions[name] = []
            for i, state in enumerate(states[name]):
                if np.random.rand() < epsilon:
                    actions[name].append(
                        np.random.randint(agent['action_size']))
                else:
                    state_tensor = torch.tensor(
                        state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_values = agent['q_network'](
                        state_tensor).detach().cpu().numpy()
                    actions[name].append(np.argmax(q_values))
        return actions

    def learn(self):
        for name, agent in self.agents.items():
            if len(agent['replay_buffer']) < agent['replay_buffer'].batch_size:
                continue

            experiences, indices, weights = agent['replay_buffer'].sample(
                beta=self.beta)
            weights = weights.to(self.device)

            states, actions, rewards, next_states, dones = zip(*experiences)
            states = torch.tensor(
                states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(
                actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.tensor(
                next_states, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32,
                                 device=self.device).unsqueeze(1)

            next_q_values = agent['target_q_network'](
                next_states).detach().max(1)[0].unsqueeze(1)
            target_q_values = rewards + \
                (self.gamma * next_q_values * (1 - dones))
            expected_q_values = agent['q_network'](states).gather(1, actions)

            td_errors = (target_q_values -
                         expected_q_values).detach().squeeze().cpu().numpy()
            agent['replay_buffer'].update_priorities(indices, td_errors)

            loss = (weights.unsqueeze(1) *
                    (expected_q_values - target_q_values) ** 2).mean()
            agent['optimizer'].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent['q_network'].parameters(), 1.0)
            agent['optimizer'].step()

            self._soft_update(agent['q_network'], agent['target_q_network'])

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1 - self.tau)
                * target_param.data)
