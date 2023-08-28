# %%
import gym
import torch
from ale_py import ALEInterface
from ale_py.roms import Pong
from pyvirtualdisplay import Display
from src.helpers import display_frames_as_gif, play_game
from src.policy_gradeint import PolicyNetwork

%matplotlib inline

# %%
# set up env
ale = ALEInterface()
ale.loadROM(Pong)
env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
env.seed(2023)
# %%
# Run a demo of the environment
display = Display(visible=0, size=(1400, 900))
display.start()
observation = env.reset()
cumulated_reward = 0

frames = []
for t in range(100):
    frames.append(env.render())
    # primitive agent, takes random actions inside the action space
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    cumulated_reward += reward
    if done:
        print("Episode finished after {} timesteps, accumulated reward = {}".format(t+1, cumulated_reward))
        break
print("Episode finished without success, accumulated reward = {}".format(cumulated_reward))

env.close()

display_frames_as_gif(frames)

# %%
# play pong with the trained agent
input_size = 80 * 80
hidden_size = 200
model = PolicyNetwork(input_size, hidden_size)
model.load_state_dict(torch.load('pg_params.pth')['model_weights'])

play_game(env, model)

