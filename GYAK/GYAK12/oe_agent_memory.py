import torch
import random
import copy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

class DQN:

    def __init__(self, network=None, output_dim=-1, gamma=0.9, batch_size=32, epsilon_decay=0.9997,
                 type_name="dqn", memory_buffer=None, update_frequency=10):
        """
        Create an additional target network for the DQN algorithm
        """
        self.back_up_network = copy.deepcopy(network)
        self.network = network
        self.update_frequency = update_frequency
        self.output = output_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.epsilon = 1
        self.type = type_name
        self.memory = memory_buffer
        self.optimizer = optim.Adam(self.network.parameters())
        self.target_network = copy.deepcopy(self.network)

    def inference(self, state):
        """
        This function if used for choosing an action with a trained model
        Strictly for Single-Agent Reinforcement Learning
        :param state: The state representation of the given environment in a given time step.
        :return: The chosen discrete action in the form of the position of the highest value of the predicted_value list.
        """
        with torch.no_grad():
            predicted_value = self.network(torch.from_numpy(state).float()).detach()
        a = np.argmax(predicted_value)
        return a

    def _get_transitions_from_batch(self):
        """
        Samples the transitions from the memory according to the given batch size
        :return: transitions as a bunch of namedtupels
        """
        return self.memory.sample(batch_size=self.batch_size)

    def _process_transitions(self):
        """
        Separates the components of the experiences
        Strictly for Discrete actions spaces
        For continuous change LongTensor to FloatTensor
        :return: The separated
        """
        experiences = self._get_transitions_from_batch()
        size = len(experiences)
        transitions = self.memory.transition(*zip(*experiences))
        state_batch = torch.FloatTensor(transitions.state)
        action_batch = torch.reshape(torch.LongTensor(transitions.action), (size, 1))
        reward_batch = torch.FloatTensor(transitions.reward)
        next_state_batch = torch.FloatTensor(transitions.next_state)
        done_batch = np.array(transitions.done)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, size

    def predict(self, state):
        """
        Strictly for Single agent Reinforcement Learning
        Exploration is done according to e-greedy policy
        :return: The chosen action in the form of a position of the actions list.
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                predicted_value = self.network(torch.from_numpy(state).float()).detach().numpy()
            a = np.argmax(predicted_value)
            return a
        else:
            a = np.random.randint(0, self.output, 1)[0]
            return a

    def fit(self):
        """
        Step 1. Get the experience components from the memory buffer.
        Step 2. Gather the Q-values of the actions that were actually chosen.
        Step 3. Calculate the target values for non-terminal transitions. If the transition was terminal than set the
        target_value equal to the gathered reward. (Bellman eq.)
        Step 4. Calculate the loss with MSE between the predicted Q values and the calculated ones.
        Step 5. Tune the network with the loss, limit the change in the parameters as suggested in Mnih et al. 2015.
        """
        self.epsilon *= self.epsilon_decay
        # STEP1.
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, size = self._process_transitions()
        # STEP2.
        predicted_q = self.network(torch.squeeze(state_batch))
        predicted_q = torch.gather(predicted_q, 1, action_batch).squeeze()
        # STEP3.
        with torch.no_grad():
            target_q = self.target_network(torch.squeeze(next_state_batch))
            target_q = target_q.max(dim=-1)[0]
        target_value = reward_batch.squeeze() + self.gamma * target_q * ~done_batch
        # STEP4.

        loss = F.mse_loss(predicted_q, target_value)
        # STEP5.
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def update_network_to_original(self):
        self.network.load_state_dict(self.back_up_network.state_dict())
