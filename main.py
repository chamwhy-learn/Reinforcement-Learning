import random
from collections import deque, defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Input

import matplotlib.pyplot as plt

import gym
from gym import spaces

if __name__ == '__main__':
    tf.keras.backend.set_float('float32')
    env = gym.make('CartPole-v0')

    obs_dim = env.observation_space.shape
    acs_dim = None

    if isinstacne