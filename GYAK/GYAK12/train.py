import torch
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

from Agents.oe_agent_memory import DQN
from Agents.uniform_memory import MemoryBuffer
from trajectorytracking.Envs.env_straight import Straight_env

# Params
Net = torch.nn.Sequential(
    torch.nn.Linear(2, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 3),
)
buffer_size = 10000
output = 3
gamma = 0.9
batch = 128                     # batch: memóriabufferekből egyszerre hány tanító mintát veszünk ki.
epsilon_decay = 0.997
update_frequency = 5
episodes = 10000                # Hányszor próbálhatja az agens megatanulni a problémát. Btw. 1.000-ből betanul

memory = MemoryBuffer(buffer_size)
agent = DQN(network=Net, output_dim=output, gamma=gamma, batch_size=batch, epsilon_decay=epsilon_decay,
            memory_buffer=memory, update_frequency=update_frequency)
env = Straight_env()

training_reward = []

for episode in range(episodes):
    state = env.reset()
    done = False
    episodic_reward = []
    for step in range(200):
        action = agent.predict(state)

        next_state, reward, done, info = env.step(action)

        agent.memory.push(state, action, reward, next_state, done)

        state = np.array(next_state)
        episodic_reward.append(reward)
        #env.render()
        if done:
            break

    if episode % update_frequency == 0:
        agent.fit()                                 # Memóriák közül kiveszünk egy batch-nyi tanítómintát
    training_reward.append(sum(episodic_reward))

    print(f"episode:{episode}, episodic_reward: {sum(episodic_reward)}")

PATH = "dqn.pt"
torch.save(agent.network.state_dict(), PATH)

plt.plot(training_reward)
plt.show()
