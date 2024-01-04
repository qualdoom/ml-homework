import torch
from network import build_model
import numpy as np
from constants import *
from torchrl.envs import *
from torchrl.envs.libs.gym import *
from torchrl.data import *
from torchrl.data.replay_buffers.samplers import PrioritizedSampler


class Agent:
    def __init__(self, num_channels, width, height, n_actions):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(num_channels, width, height, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)
        self.rb = TensorDictReplayBuffer(storage=ListStorage(SIZE), sampler=PrioritizedSampler(SIZE, alpha=0.6, beta=0.8), priority_key="td_error")
        self.n_actions = n_actions

    def select_action(self, state, epsilon=0):
        # assume that state is pixels
        if np.random.rand() < epsilon:
            return np.random.choice(np.arange(0, self.n_actions))
        
        value, index = torch.max(self.model(state).detach(), 1)

        return index
    
    def calculate_td_error(self, state, action, reward, next_state, done, gamma=GAMMA):
        action = torch.max(action, 0)[1]
        predict_qvalue = (self.model(state).to(self.device))[0][action].detach()

        predict_next_qvalue = (self.model(next_state).detach().to(self.device))

        next_state_value = torch.max(predict_next_qvalue, 1)[0].to(self.device)

        target_qvalue = (reward + gamma * next_state_value)

        if done:
            target_qvalue = reward

        td_error = abs(target_qvalue - predict_qvalue)

        return td_error
    
    def compute_each_loss(self, states, _actions, rewards, next_states, is_done, gamma=GAMMA):
        actions = torch.max(_actions, 1)[1].to(self.device)

        predicted_qvalues = self.model(states).to(self.device)

        predicted_qvalues_for_actions = predicted_qvalues[
          range(states.shape[0]), actions
        ].to(self.device)

        predicted_next_qvalues = self.model(next_states).detach().to(self.device)

        next_state_values = torch.max(predicted_next_qvalues, 1)[0].to(self.device)

        target_qvalues_for_actions = (rewards + gamma * next_state_values).to(self.device)

        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions).to(self.device)
        
        priority = (abs(predicted_qvalues_for_actions - target_qvalues_for_actions.detach())).detach().to(self.device)

        loss = ((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2).to(self.device)
        
        return loss, priority
      
    
    def record_experience(self, state): # state is a tensor
        state['td_error'] = torch.as_tensor(self.calculate_td_error(state['pixels_trsf'], state['action'], state['next']['reward'], state['next']['pixels_trsf'], state['next']['done'])) + 1e-6

        state['pixels_trsf'] = state['pixels_trsf'].unsqueeze(0)
        state['next'] = state['next'].unsqueeze(0)
        state['action'] = state['action'].unsqueeze(0)

        state.batch_size = [1]

        self.rb.extend(state)


    def train(self, batch_size=1000):
        batch= self.rb.sample(batch_size)

        self.optimizer.zero_grad()

        element_wise_loss, priority = self.compute_each_loss(batch['pixels_trsf'], batch['action'], batch['next']['reward'], batch['next']['pixels_trsf'], batch['next']['done'])

        value = (element_wise_loss * batch['_weight'] * batch['_weight']).detach()

        batch['td_error'] = priority + 1e-6

        loss = torch.mean(element_wise_loss * batch['_weight'] * batch['_weight'])

        loss.backward()

        self.optimizer.step()

        self.rb.update_tensordict_priority(batch)

        return value.cpu().numpy().mean()
        