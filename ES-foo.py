# Evolution Strategies on foo-v0
# https://blog.openai.com/evolution-strategies/

import gym
import gym_foo
from gym import wrappers
import numpy as np

env = wrappers.Monitor(gym.make('foo-v0'), 'ES-foo-v0', force=True)
print('------------')
print('env.observation_space', env.observation_space)
print('env.action_space', env.action_space)
print('------------')
print ('env.action_space.n = %d' % (env.action_space.n))
print('The Discrete space allows a fixed range of non-negative numbers, so in this case valid actions are either 0 or 1')
print('------------')
print ('env.observation_space.shape[0] = %d' % (env.observation_space.shape[0]))
print('env.observation_space = [position of cart, velocity of cart, angle of pole, rotation rate of pole]')
print('env.observation_space.high : {:}'.format(env.observation_space.high))
print('env.observation_space.low : {:}'.format(env.observation_space.low))

np.random.seed(0)

# hyperparameters

npop = 50   # population size
sigma = 0.15    # noise standard deviation
alpha = 0.15    # learning rate

print('------------')
print('npop = %d, sigma = %.2f, alpha = %.2f' % (npop, sigma, alpha))

# ... 1) create a neural network with weights w

w = np.random.randn(env.observation_space.shape[0])     # our initial guess is random

print('------------')
print('w : {:}'.format(w))

def get_action(state, w):
    def sigmoid(x):
      return 1 / (1 + np.exp(-x))
    action = np.tanh((np.dot(state.T, w)))
    return 0 if action < 0 else 1

# The black-box function (f) we want to optimize using
# Natural Evolution Strategies (NES), where the parameter distribution
# is a gaussian of fixed standard deviation

# ... 2) run the neural network on the environment for some time
# ... 3) sum up and return the total reward

def f(w, render=False):
    state = env.reset()
    total_reward = 0
    for t in range(1000):
        if render: env.render()
        action = get_action(state, w)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

for i in range(50):
    # initialize memory for a population of w's, and their rewards
    N = np.random.randn(npop, env.observation_space.shape[0])   # samples from a normal distribution
    R = np.zeros(npop)
    for j in range(npop):
        w_try = w + sigma*N[j]      # jitter w using gaussian of sigma 0.1
        R[j] = f(w_try)     # evaluate the jittered version
    
    # standardize the rewards to have a gaussian distribution
    A = (R - np.mean(R)) / (np.std(R)+1e-6)

    # Perform the parameter update. The matrix multiply below
    # is just an efficient way to sum up all the rows of the noise matrix N,
    # where each row N[j] is weighted by A[j]
    w = w + alpha/(npop*sigma) * np.dot(N.T, A)

    # print current fitness of the most likely parameter setting
    print('iter %d, f(w) = %.2f' % (i, f(w)))
    print('w : {:}'.format(w))

env.close()

# Reference : https://github.com/openai/gym/wiki/CartPole-v0
