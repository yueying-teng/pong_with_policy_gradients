"""
trains a Pong agent using (stochastic) Policy Gradients and OpenAI Gym.
reference https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

import datetime
import os
import random
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from src.helpers import preprocess


# labels of the moving up and down in Pong
UP = 2
DOWN = 3

is_cuda = torch.cuda.is_available()


def calc_discounted_future_rewards(rewards, discount_factor):
    """
    Calculate discounted future reward at each timestep.

    discounted_future_reward[t] = \sum_{k=1} discount_factor^k * reward[t+k]
    """

    discounted_future_rewards = torch.empty(len(rewards))

    discounted_future_reward = 0
    for t in range(len(rewards) - 1, -1, -1):
        # If rewards[t] != 0, we are at game boundary (win or loss) so we
        # reset discounted_future_reward to 0 (this is pong specific!)
        if rewards[t] != 0:
            discounted_future_reward = 0

        discounted_future_reward = rewards[t] + discount_factor * discounted_future_reward
        discounted_future_rewards[t] = discounted_future_reward

    return discounted_future_rewards


class PolicyNetwork(nn.Module):
    """
    Simple two-layer MLP as the policy network.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        prob_up = torch.sigmoid(x)

        return prob_up


def run_episode(model, env, discount_factor, render=False):
    """
    Run the game for one episode.
    """
    observation = env.reset()
    observation = observation[0]
    prev_x = preprocess(observation)

    action_chosen_log_probs = []
    rewards = []

    done = False
    timestep = 0

    while not done:
        if render:
            # Render the game window at 30fps
            time.sleep(1 / 30)
            env.render()

        # Input to the network is the difference between two frames
        cur_x = preprocess(observation)
        x = cur_x - prev_x
        prev_x = cur_x

        # Run the policy network and sample action from the returned probability
        prob_up = model(x)
        action = UP if random.random() < prob_up else DOWN # roll the dice!

        # Calculate the probability of the network sampling the chosen action
        action_chosen_prob = prob_up if action == UP else (1 - prob_up)
        action_chosen_log_probs.append(torch.log(action_chosen_prob))

        # Make one move using the chosen action to get the new measurements,
        # and updated discounted_reward
        _, reward, done, _, _ = env.step(action)
        rewards.append(torch.Tensor([reward]))
        timestep += 1

    # Concat lists of log probs and rewards into 1-D tensors
    action_chosen_log_probs = torch.cat(action_chosen_log_probs)
    rewards = torch.cat(rewards)

    # Calculate the discounted future reward at each timestep
    discounted_future_rewards = calc_discounted_future_rewards(rewards, discount_factor)

    # Standardize the rewards to have mean 0, std. deviation 1 (helps control the gradient estimator variance).
    # It encourages roughly half of the actions to be rewarded and half to be discouraged, which
    # is helpful especially in the beginning when positive reward signals are rare.
    discounted_future_rewards = (discounted_future_rewards - discounted_future_rewards.mean()) \
                                     / discounted_future_rewards.std()

    # PG magic happens right here, multiplying action_chosen_log_probs by future reward.
    # Negate since the optimizer does gradient descent (instead of gradient ascent)
    loss = -(discounted_future_rewards * action_chosen_log_probs).sum()

    return loss, rewards.sum()


def train(env, model, learning_rate, discount_factor, batch_size, save_every_batches, render=False):
    # Load model weights and metadata from checkpoint if exists
    if os.path.exists('pg_params.pth'):
        print('Loading from checkpoint...')
        save_dict = torch.load('pg_params.pth')

        model.load_state_dict(save_dict['model_weights'])
        start_time = save_dict['start_time']
        last_batch = save_dict['last_batch']
    else:
        start_time = datetime.datetime.now().strftime('%H.%M.%S-%m.%d.%Y')
        last_batch = -1

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    # Set up tensorboard logging
    writer = SummaryWriter(log_dir='tensorboard_logs', filename_suffix=start_time)

    # Pick up at the batch number we left off at to make tensorboard plots nicer
    batch = last_batch + 1
    while True:
        mean_batch_loss = 0
        mean_batch_reward = 0
        for _ in range(batch_size):
            # Run one episode
            loss, episode_reward = run_episode(model, env, discount_factor, render)
            mean_batch_loss += loss / batch_size
            mean_batch_reward += episode_reward / batch_size
            print(f'Episode reward total was {episode_reward}')

        # Backprop after `batch_size` episodes
        optimizer.zero_grad()
        mean_batch_loss.backward()
        optimizer.step()

        # Batch metrics and tensorboard logging
        print(f'Batch: {batch}, mean loss: {mean_batch_loss:.2f}, '
              f'mean reward: {mean_batch_reward:.2f}')
        writer.add_scalar('mean_loss', mean_batch_loss.detach().item(), global_step=batch)
        writer.add_scalar('mean_reward', mean_batch_reward.detach().item(), global_step=batch)

        if batch % save_every_batches == 0:
            print('Saving checkpoint...')
            save_dict = {
                'model_weights': model.state_dict(),
                'start_time': start_time,
                'last_batch': batch
            }
            torch.save(save_dict, 'pg_params.pth')

        batch += 1
