# %%
import gym
from ale_py import ALEInterface
from ale_py.roms import Pong
from src.policy_gradeint import PolicyNetwork, train

# %%
discount_factor = 0.99 # discount factor for reward

batch_size = 10
save_every_batches = 5
# learning_rate = 7e-4
learning_rate = 1e-3

input_size = 80 * 80 # input dimensionality: 80x80 grid
hidden_size = 200 # number of hidden layer neurons
model = PolicyNetwork(input_size, hidden_size)

ale = ALEInterface()
ale.loadROM(Pong)
env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
env.seed(2023)

# %%
train(
    env,
    model,
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    batch_size=batch_size,
    save_every_batches=save_every_batches,
    render=False,
    )
