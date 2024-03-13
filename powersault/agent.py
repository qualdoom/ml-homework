import torch
from network import NeuralNetwork
import numpy as np
from constants import *
from torchrl.envs import *
from torchrl.envs.libs.gym import *
from torchrl.data import *
from torchrl.data.replay_buffers.samplers import PrioritizedSampler


class Agent:
    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def __init__(self, num_channels, width, height, n_actions):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = NeuralNetwork(num_channels=num_channels, height=height, width=width,
                                            n_actions=n_actions).to(self.device)  # init policy network
        self.count_until_change_target_network = 0
        self.target_network = NeuralNetwork(num_channels=num_channels, height=height, width=width,
                                            n_actions=n_actions).to(self.device)
        self.update_target_network()  # init target network (equals to policy_network)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)
        self.rb = TensorDictReplayBuffer(storage=ListStorage(SIZE),
                                         sampler=PrioritizedSampler(SIZE, alpha=0.6, beta=0.8), priority_key="td_error")
        self.n_actions = n_actions
        self.frames = []
        self.loss_history = []
        self.t_max = 10000
        self.reward_history = []
        self.epsilon = 0.7
        self.count_for_learning = 0

    def select_action(self, state, epsilon=0):
        # assume that state is pixels
        if np.random.rand() < epsilon:
            return np.random.choice(np.arange(0, self.n_actions))

        value, index = torch.max(self.policy_network(state).detach(), 1)

        return index

    def calculate_td_error(self, state, action, reward, next_state, done, gamma=GAMMA):
        action = torch.max(action, 0)[1]
        predict_q_value = (self.policy_network(state).to(self.device))[0][action].detach()

        predict_next_q_value = (self.target_network(next_state).detach().to(self.device))

        next_state_value = torch.max(predict_next_q_value, 1)[0].to(self.device)

        target_q_value = (reward + gamma * next_state_value)

        if done:
            target_q_value = reward

        td_error = abs(target_q_value - predict_q_value)

        return td_error

    def compute_each_loss(self, states, _actions, rewards, next_states, is_done, gamma=GAMMA):
        actions = torch.max(_actions, 1)[1].to(self.device)

        predicted_q_values = self.policy_network(states).to(self.device)

        predicted_q_values_for_actions = predicted_q_values[
            range(states.shape[0]), actions
        ].to(self.device)

        predicted_next_q_values = self.target_network(next_states).detach().to(self.device)

        next_state_values = torch.max(predicted_next_q_values, 1)[0].to(self.device)

        target_q_values_for_actions = (rewards + gamma * next_state_values).to(self.device)

        target_q_values_for_actions = torch.where(
            is_done, rewards, target_q_values_for_actions).to(self.device)

        priority = (abs(predicted_q_values_for_actions - target_q_values_for_actions.detach())).detach().to(self.device)
        
        loss_function = torch.nn.HuberLoss(delta=1.0)
        loss = loss_function(predicted_q_values_for_actions, target_q_values_for_actions).to(self.device)
        
        return loss, priority

    def record_experience(self, state):  # state is a tensor
        if self.count_until_change_target_network == FRAMES_FOR_UPDATE_TARGET:
            self.update_target_network()
            self.count_until_change_target_network = 0

        state['td_error'] = torch.as_tensor(
            self.calculate_td_error(state['pixels_trsf'], state['action'], state['next']['reward'],
                                    state['next']['pixels_trsf'], state['next']['done'])) + 1e-6

        state['pixels_trsf'] = state['pixels_trsf'].unsqueeze(0)
        state['next'] = state['next'].unsqueeze(0)
        state['action'] = state['action'].unsqueeze(0)
        
        if self.count_for_learning == FRAMES_FOR_UPDATE_NETWORK:
            self.train(16)
            self.count_for_learning = 0

        state.batch_size = [1]

        self.rb.extend(state)

        self.count_until_change_target_network += 1

    def train(self, batch_size=1000):
        batch = self.rb.sample(batch_size)

        self.optimizer.zero_grad()

        element_wise_loss, priority = self.compute_each_loss(batch['pixels_trsf'], batch['action'],
                                                             batch['next']['reward'], batch['next']['pixels_trsf'],
                                                             batch['next']['done'])

        value = (element_wise_loss * batch['_weight']).detach()

        batch['td_error'] = priority + 1e-6

        loss = torch.mean(element_wise_loss * batch['_weight'])

        loss.backward()

        self.optimizer.step()

        self.rb.update_tensordict_priority(batch)

        return value.cpu().numpy().mean()

    def load(self):
        checkpoint = torch.load('model.pth', map_location=torch.device(self.device))
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.count_until_change_target_network = checkpoint['count_until_change_target_network']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.frames = checkpoint['frames']
        self.loss_history = checkpoint['loss_history']
        self.t_max = checkpoint['t_max']
        self.reward_history = checkpoint['reward_history']
        self.epsilon = checkpoint['epsilon']

    def save(self):
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'count_until_change_target_network': self.count_until_change_target_network,
            'optimizer': self.optimizer.state_dict(),
            'frames': self.frames,
            'loss_history': self.loss_history,
            't_max': self.t_max,
            'reward_history': self.reward_history,
            'epsilon': self.epsilon
        }, 'model.pth')

