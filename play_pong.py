import gym
import torch
from ale_py import ALEInterface
from ale_py.roms import Pong
from src.helpers import save_frames_as_gif, play_game
from src.policy_gradeint import PolicyNetwork


# set up env
ale = ALEInterface()
ale.loadROM(Pong)
env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
# print("List of available actions: ", env.unwrapped.get_action_meanings())
env.seed(2023)

# Run a demo of the environment
observation = env.reset()
cumulated_reward = 0

frames = []
for t in range(10000):
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

save_frames_as_gif(frames, fn='resources/demo_pong')

# play pong with the trained agent
input_size = 80 * 80
hidden_size = 200
model = PolicyNetwork(input_size, hidden_size)
model.load_state_dict(torch.load('pg_params.pth')['model_weights'])

play_game(env, model, fn='resources/play_pong')

