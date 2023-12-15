from Model import NeuralNetwork
import torch

from functorch import vmap
import torchvision.transforms as transforms 
import numpy as np

# from stable_baselines3 import HerReplayBuffer
# from stable_baselines.common.buffers import PrioritizedReplayBuffer
    
def get_img_as_tensor(state):
    # изображение как тензор в torch
    transform = transforms.ToTensor()
    tensor = transform(state)
    return tensor

class Agent:
    
    saving_path = "models.txt"
    learning_rate = 5e-3
    epsilon = 0.2
    
    def build_model(self, num_channels, height, width, n_actions):
        model = NeuralNetwork(num_channels=num_channels, height=height, width=width, n_actions=n_actions)
        self.n_actions = n_actions
        # self.buffer = PrioritizedReplayBuffer(50000)
        return model

    def __init__(self, num_channels, height, width, n_actions):
        self.model = self.build_model(num_channels=num_channels, height=height, width=width, n_actions=n_actions)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load(self):
        self.model.load_state_dict(torch.load(self.saving_path))
        self.model.eval()
        f = open('epsilon.txt', 'r')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
        try:
            self.epsilon = (float)(f.readline())
        except:
            self.epsilon = 0.2
        f.close()    

    def save(self):
        # save model parameters
        torch.save(self.model.state_dict(), self.saving_path)
    
        f = open('epsilon.txt','w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
        f.write('{}'.format(self.epsilon))
        f.close()

    def get_action(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.choice(np.arange(0, self.n_actions))

        state = get_img_as_tensor(state).unsqueeze(0)
        q_values = self.model(state).detach().numpy()

        return int(np.argmax(q_values))

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99, check_shapes=False):
        """ Compute td loss using torch operations only. Use the formula above. """
        states = torch.tensor(states, dtype=torch.float32)    # shape: [batch_size, state_size]
        actions = torch.tensor(actions, dtype=torch.long)    # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [batch_size]
        next_states = torch.tensor(next_states, dtype=torch.float32) # shape: [batch_size, state_size]
        is_done = torch.tensor(is_done, dtype=torch.uint8)  # shape: [batch_size]

        # get q-values for all actions in current states
        predicted_qvalues = self.model(states)

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
          range(states.shape[0]), actions
        ]

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.model(next_states).detach()

        # compute V*(next_states) using predicted next q-values
        # print(predicted_next_qvalues)
        next_state_values = torch.max(predicted_next_qvalues, 1)[0]
        # print(next_state_values)

        assert next_state_values.dtype == torch.float32

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + gamma * next_state_values

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        loss = torch.mean((predicted_qvalues_for_actions -
                          target_qvalues_for_actions.detach()) ** 2)

        if check_shapes:
            assert predicted_next_qvalues.data.dim(
            ) == 2, "make sure you predicted q-values for all actions in next state"
            assert next_state_values.data.dim(
            ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
            assert target_qvalues_for_actions.data.dim(
            ) == 1, "there's something wrong with target q-values, they must be a vector"

        return loss

    def step(self, states, actions, rewards, next_states, dones):
        self.opt.zero_grad()
        L = self.compute_td_loss(states, actions, rewards, next_states, dones)
        x = L
        L.backward()
        self.opt.step() 

        return x