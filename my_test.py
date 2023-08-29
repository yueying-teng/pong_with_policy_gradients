import os
import argparse
import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


is_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch policy gradient example at openai-gym pong')
gamma = 0.99 # discount factor of rewards
decay_rate = 0.99
learning_rate = 1e-3
batch_size = 10
seed = 87
test = True # whether to test the trained model or keep training

env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
env.seed(seed)
torch.manual_seed(seed)


D = 80 * 80

if test == True:
    render = True
else:
    render = False

def prepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0 ] = 1
    return I.astype(float).ravel()


class Policy(nn.Module):
    def __init__(self, num_actions=2):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.affine2 = nn.Linear(200, 200)
        self.affine3 = nn.Linear(200, num_actions) # action 1: static, action 2: move up, action 3: move down

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.affine2(x)
        action_scores = self.affine3(x)
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

# built policy network
policy = Policy()
if is_cuda:
    policy.cuda()

# # check & load pretrain model
# if os.path.isfile('pg_params.pkl'):
#     print('Load Policy Network parametets ...')
#     policy.load_state_dict(torch.load('pg_params.pkl'))


optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate, weight_decay=decay_rate)


# Main loop
running_reward = None
reward_sum = 0
prev_x = None

mean_batch_loss = 0
mean_batch_reward = 0

for episode_number in count(1):
    # run one episode
    state = env.reset()
    state = state[0]
    for t in range(10000):
        # if render:
        #     env.render()

        cur_x = prepro(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        action = policy.select_action(x)
        action_env = action + 2
        # state, reward, done, _ = env.step(action_env)
        state, reward, done, truncated, info = env.step(action_env)
        reward_sum += reward

        policy.rewards.append(reward)

        if done:
            # episode_number += 1
            # tracking log
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('REINFORCE ep %03d done. reward: %f. reward running mean: %f' % (episode_number, reward_sum, running_reward))
            reward_sum = 0
            break

    # discounted future rewards for one episode
    R = 0
    policy_loss = []
    episode_reward = []
    for r in policy.rewards[::-1]:
        R = r + 0.99 * R
        episode_reward.insert(0, R)
    # turn rewards to pytorch tensor and standardize
    episode_reward = torch.Tensor(episode_reward)
    episode_reward = (episode_reward - episode_reward.mean()) / (episode_reward.std() + 1e-6)

    for log_prob, reward in zip(policy.saved_log_probs, episode_reward):
        policy_loss.append(- log_prob * reward)
    policy_loss = torch.stack(policy_loss).sum()

    episode_reward_sum = sum(policy.rewards)

    del policy.rewards[:]
    del policy.saved_log_probs[:]

    mean_batch_loss += policy_loss / batch_size
    mean_batch_reward += episode_reward_sum / batch_size

    if episode_number % batch_size == 0:
        # use policy gradient update model weights
        # Backprop after `batch_size` episodes
        optimizer.zero_grad()
        if is_cuda:
            mean_batch_loss.cuda()
        mean_batch_loss.backward()
        optimizer.step()

        print(f'at episode_number {episode_number}')
        print('mean_loss', mean_batch_loss.detach().item())
        print('mean_reward', mean_batch_reward)
        mean_batch_loss = 0
        mean_batch_reward = 0

    # Save model in every 50 episode
    if episode_number % 50 == 0:
        print('ep %d: model saving...' % (episode_number))
        torch.save(policy.state_dict(), 'pg_params.pkl')

        # if episode_number == total_episodes:
        #     break
