import torch
from network import NeuralNetwork
import numpy as np
from constants import *
from torchrl.envs import *
from torchrl.envs.libs.gym import *
from torchrl.data import *
from torchrl.data.replay_buffers.samplers import PrioritizedSampler


class Agent:
    def __init__(self, num_channels, width, height, n_actions): # initialize Agent
        self.device = self.set_device()
        self.num_channels = num_channels
        self.width = width
        self.height = height
        self.n_actions = n_actions
        self.initialize_components()

    def set_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_components(self):
        self.setup_networks()
        self.setup_optimizer()
        self.setup_replay_buffer()
        self.setup_agent_properties()

    def setup_networks(self):
        self.policy_network = self.create_network().to(self.device)
        self.target_network = self.create_network().to(self.device)
        self.sync_networks()

    def create_network(self):
        return NeuralNetwork(num_channels=self.num_channels, height=self.height, width=self.width,
                             n_actions=self.n_actions)

    def sync_networks(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)

    def setup_replay_buffer(self):
        self.replay_buffer = TensorDictReplayBuffer(storage=ListStorage(SIZE),
                                                    sampler=PrioritizedSampler(SIZE, alpha=0.6, beta=0.8),
                                                    priority_key="td_error")

    def setup_agent_properties(self):
        self.frames = []
        self.loss_history = []
        self.reward_history = []
        self.epsilon = 0.7
        self.t_max = 10000
        self.reset_counters()

    def reset_counters(self):
        self.count_until_change_target_network = 0
        self.count_for_learning = 0

    def select_action(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.choice(np.arange(self.n_actions))
        value, index = torch.max(self.policy_network(state).detach(), 1)
        return index.item()

    def calculate_td_error(self, state, action, reward, next_state, done, gamma=GAMMA):
        with torch.no_grad():
            current_q_value = self.policy_network(state)[0][action]
            next_q_value = self.target_network(next_state).max(1)[0]
            target_q_value = reward + (gamma * next_q_value * (1 - int(done)))
            td_error = abs(target_q_value - current_q_value)
        return td_error

    def compute_loss(self, states, actions, rewards, next_states, dones, gamma=GAMMA):
        states, actions, rewards, next_states, dones = self.convert_to_tensors(states, actions, rewards, next_states, dones)
        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_network(next_states).max(1)[0]
        expected_q_values = rewards + gamma * next_q_values * (1 - dones)
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, expected_q_values)
        return loss

    def convert_to_tensors(self, states, actions, rewards, next_states, dones):
        print(states, actions, rewards, next_states, dones)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        return states, actions, rewards, next_states, dones

    def record_experience(self, state):
        self.count_until_change_target_network += 1
        if self.count_until_change_target_network >= FRAMES_FOR_UPDATE_TARGET:
            self.sync_networks()
            self.reset_counters()

        td_error = self.calculate_td_error(**state) + 1e-6
        state['td_error'] = td_error
        self.replay_buffer.extend(state)

    def train(self, batch_size=1000):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(**batch)
        self.perform_backpropagation(loss)
        self.update_priorities(batch)
        return loss.item()

    def perform_backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_priorities(self, batch):
        priorities = batch['td_error'] + 1e-6
        self.replay_buffer.update_tensordict_priority(batch, priorities)

    def load(self, filepath='model.pth'):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_networks_and_optimizer(checkpoint)
        self.load_agent_state(checkpoint)

    def load_networks_and_optimizer(self, checkpoint):
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def load_agent_state(self, checkpoint):
        self.count_until_change_target_network = checkpoint.get('count_until_change_target_network', 0)
        self.frames = checkpoint.get('frames', [])
        self.loss_history = checkpoint.get('loss_history', [])
        self.t_max = checkpoint.get('t_max', 10000)
        self.reward_history = checkpoint.get('reward_history', [])
        self.epsilon = checkpoint.get('epsilon', 0.7)

    def save(self, filepath='model.pth'):
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'count_until_change_target_network': self.count_until_change_target_network,
            'frames': self.frames,
            'loss_history': self.loss_history,
            't_max': self.t_max,
            'reward_history': self.reward_history,
            'epsilon': self.epsilon
        }, filepath)
