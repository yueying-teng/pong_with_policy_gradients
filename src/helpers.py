import random
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


random.seed(2023)
np.random.seed(2023)

def save_frames_as_gif(frames, fn):
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
    anim.save(f'{fn}.gif')


def preprocess(image):
    """
    Pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
    """

    # Crop, downsample by factor of 2, and turn to grayscale by keeping only red channel
    image = image[35: 195]
    image = image[::2, ::2, 0]

    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1

    return image.astype(float).ravel()


def model_step(model, observation, prev_x):
    # preprocess the observation, set input to network to be difference image
    cur_x = preprocess(observation)
    x = cur_x - prev_x
    prev_x = cur_x

    # run the policy network and sample an action from the returned probability
    action = model.select_action(x)
    action_env = action + 2

    return action_env, prev_x


def play_game(env, model, fn):
    observation = env.reset()
    observation = observation[0]

    prev_x = preprocess(observation) # at the beginning of the game, cur_x and prev_x are identical

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
    save_frames_as_gif(frames, fn)

