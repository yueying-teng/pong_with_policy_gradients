"""
trains a Pong agent using (stochastic) Policy Gradients and OpenAI Gym.
reference https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

import datetime
import os
import numpy as np
from itertools import count

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from src.helpers import preprocess


is_cuda = torch.cuda.is_available()
torch.manual_seed(2023)


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions=2):
        super(PolicyNetwork, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions) # action 1: static, action 2: move up, action 3: move down

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_scores = self.fc3(x)
        return F.softmax(action_scores, dim=1)

    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        if is_cuda:
            x = x.cuda()
        probs = self.forward(x)

        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append(m.log_prob(action))
        return action


def run_episode(model, env, episode_number, prev_x, running_reward):
    """
    Run the game for one episode.
    """
    reward_sum = 0
    observation = env.reset()[0]

    for _ in range(10000):
        cur_x = preprocess(observation)
        # Input to the network is the difference between two frames
        x = cur_x - prev_x if prev_x is not None else np.zeros(model.input_size)

        prev_x = cur_x

        # Run the policy network and sample action from the returned probability
        # During training, the agent needs to balance exploration (trying new actions)
        # and exploitation (choosing actions based on the current policy).
        action = model.select_action(x)
        action_env = action + 2
        # Make one move using the chosen action to get the new measurements,
        # and updated discounted_reward
        observation, reward, done, _, _ = env.step(action_env)
        reward_sum += reward

        model.rewards.append(reward)

        if done:
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('REINFORCE ep %03d done. reward: %f. reward running mean: %f' % (episode_number, reward_sum, running_reward))
            break

    return running_reward, prev_x


def calc_discounted_future_rewards(model, discount_factor):
    """
    Calculate discounted future reward at each timestep.

    discounted_future_reward[t] = \sum_{k=1} discount_factor^k * reward[t+k]
    """
    R = 0
    policy_loss = []
    # Discounted future reward at EACH timestep
    episode_reward = []
    for r in model.rewards[::-1]:
        R = r + discount_factor * R
        episode_reward.insert(0, R)
    # Turn rewards to pytorch tensor and standardize
    episode_reward = torch.Tensor(episode_reward)
    # Standardize the rewards to have mean 0, std. deviation 1 (helps control the variance of the gradient estimator).
    episode_reward = (episode_reward - episode_reward.mean()) / (episode_reward.std() + 1e-6)

    # Policy gradient magic happens here, multiplying saved_log_probs by discounted future reward.
    # Negate since the optimizer does gradient descent (instead of gradient ascent)
    for log_prob, reward in zip(model.saved_log_probs, episode_reward):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.stack(policy_loss).sum()

    return policy_loss


def train(env, model, learning_rate, discount_factor, batch_size, save_every_episodes, weight_decay, total_episodes):
    if is_cuda:
        model.cuda()

    # Load model weights and metadata from checkpoint if exists
    if os.path.exists('pg_params.pth'):
        print('Loading from checkpoint...')
        save_dict = torch.load('pg_params.pth')
        model.load_state_dict(save_dict['model_weights'])
        start_time = save_dict['start_time']
    else:
        start_time = datetime.datetime.now().strftime('%H.%M.%S-%m.%d.%Y')

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set up tensorboard logging
    writer = SummaryWriter(log_dir='tensorboard_logs', filename_suffix=start_time)

    running_reward = None
    prev_x = None
    mean_batch_loss = 0
    mean_batch_reward = 0

    for episode_number in count(1):
        # Run one episode
        running_reward, prev_x = run_episode(model, env, episode_number, prev_x=prev_x, running_reward=running_reward)

        # Discounted future rewards for one episode
        policy_loss = calc_discounted_future_rewards(model, discount_factor)
        episode_reward_sum = sum(model.rewards)
        del model.rewards[:]
        del model.saved_log_probs[:]

        mean_batch_loss += policy_loss / batch_size
        mean_batch_reward += episode_reward_sum / batch_size

        # Backprop after `batch_size` episodes
        if episode_number % batch_size == 0:
            optimizer.zero_grad()
            if is_cuda:
                mean_batch_loss.cuda()
            mean_batch_loss.backward()
            optimizer.step()

            print(f'at episode_number {episode_number}')
            print('mean_loss', mean_batch_loss.detach().item())
            print('mean_reward', mean_batch_reward)
            writer.add_scalar('mean_loss', mean_batch_loss.detach().item(), global_step=episode_number)
            writer.add_scalar('mean_reward', mean_batch_reward, global_step=episode_number)

            mean_batch_loss = 0
            mean_batch_reward = 0

        # Save model in every save_every_episodes episode
        if episode_number % save_every_episodes == 0:
            print('ep %d: model saving...' % (episode_number))
            save_dict = {
                'model_weights': model.state_dict(),
                'start_time': start_time,
            }
            torch.save(save_dict, 'pg_params.pth')

        if episode_number == total_episodes:
            break
