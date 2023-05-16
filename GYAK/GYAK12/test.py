import torch
from Agents.agents.DQN_sampling import DQN
from Agents.agents.DQN_sampling import MemoryBuffer
from trajectorytracking.Envs.env_straight import Straight_env

PATH = "dqn.pt"

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
batch = 128
epsilon_decay = 0.997
update_frequency = 5
episodes = 1000

memory = MemoryBuffer(buffer_size)
agent = DQN(network=Net, output_dim=output, gamma=gamma, batch_size=batch, epsilon_decay=epsilon_decay,
            memory_buffer=memory, update_frequency=update_frequency)
agent.network.load_state_dict(torch.load(PATH))
agent.network.eval()
env = Straight_env()

for j in range(100):
    state = env.reset()
    for i in range(100):
        action = agent.inference(state)
        env.render()
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break