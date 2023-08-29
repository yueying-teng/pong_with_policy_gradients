import gym
from ale_py import ALEInterface
from ale_py.roms import Pong

from src.policy_gradeint import PolicyNetwork, train


discount_factor = 0.99 # discount factor for reward
total_episodes = 6000
batch_size = 10
save_every_episodes = 50
learning_rate = 1e-3
weight_decay=0.99

input_size = 80 * 80 # input dimensionality: 80x80 grid
hidden_size = 200 # number of hidden layer neurons

# set up env
ale = ALEInterface()
ale.loadROM(Pong)
env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
env.seed(2023)

model = PolicyNetwork(input_size, hidden_size)

train(
    env,
    model,
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    batch_size=batch_size,
    save_every_episodes=save_every_episodes,
    weight_decay=weight_decay,
    total_episodes=total_episodes
    )
