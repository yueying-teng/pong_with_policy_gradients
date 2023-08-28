import random
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import torch


random.seed(2023)
np.random.seed(2023)
# labels of the moving up and down in Pong
UP = 2
DOWN = 3

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 144)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    plt.close(anim._fig)
    display(HTML(anim.to_jshtml()))


def preprocess(image):
    """
    Pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
    """

    image = torch.Tensor(image)

    # Crop, downsample by factor of 2, and turn to grayscale by keeping only red channel
    image = image[35: 195]
    image = image[::2, ::2, 0]

    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1

    return image.flatten().float()


def model_step(model, observation, prev_x):
    # preprocess the observation, set input to network to be difference image
    cur_x = preprocess(observation)
    x = cur_x - prev_x
    prev_x = cur_x

    # run the policy network and sample an action from the returned probability
    prob_up = model(x)
    action = UP if random.random() < prob_up else DOWN # roll the dice!

    return action, prev_x


def play_game(env, model):
    observation = env.reset()
    observation = observation[0]
    prev_x = preprocess(observation) # at the beginning of the game, cur_x and prev_x are idential

    frames = []
    cumulated_reward = 0

    for t in range(1000):
        frames.append(env.render())
        action, prev_x = model_step(model, observation, prev_x)
        observation, reward, done, _, _ = env.step(action)
        cumulated_reward += reward
        if done:
            print("Episode finished after {} timesteps, accumulated reward = {}".format(t + 1, cumulated_reward))
            break

    print("Episode finished without success, accumulated reward = {}".format(cumulated_reward))
    env.close()
    display_frames_as_gif(frames)

