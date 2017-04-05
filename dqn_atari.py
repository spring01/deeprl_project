#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import sys
import random
import subprocess

import numpy as np
import tensorflow as tf
from keras.layers import (Conv2D, Dense, Flatten, Input,
                          Lambda, Merge, merge)
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam

from dqn.dqn import DQNAgent
from dqn.objectives import mean_huber_loss, null_loss
from dqn.preprocessors import AtariPreprocessor
from dqn.policy import *
from dqn.memory import ReplayMemory

import gym
from gym import wrappers
import cPickle as pickle


def create_model(window, input_shape, num_actions, model_name='dqn'):
    model_input_shape = tuple([window] + list(input_shape))
    state = Input(shape=model_input_shape)
    conv1 = Conv2D(32, 8, 8, subsample=(4, 4),
        border_mode='same', activation='relu', init='uniform')(state)
    conv2 = Conv2D(64, 4, 4, subsample=(2, 2),
        border_mode='same', activation='relu', init='uniform')(conv1)
    conv3 = Conv2D(64, 3, 3, subsample=(1, 1),
        border_mode='same', activation='relu', init='uniform')(conv2)
    feature = Flatten()(conv3)
    if model_name == 'dqn':
        hid = Dense(512, activation='relu', init='uniform')(feature)
        q_value = Dense(num_actions, init='uniform')(hid)
    elif model_name == 'dueling_dqn':
        value1 = Dense(512, activation='relu', init='uniform')(feature)
        value2 = Dense(1)(value1)
        advantage1 = Dense(512, activation='relu', init='uniform')(feature)
        advantage2 = Dense(num_actions, init='uniform')(advantage1)
        mean_advantage2 = Lambda(lambda x: K.mean(x, axis=1))(advantage2)
        ones = K.ones([1, num_actions])
        exp_mean_advantage2 = Lambda(lambda x: K.dot(K.expand_dims(x, dim=1), -ones))(mean_advantage2)
        sum_adv = merge([exp_mean_advantage2, advantage2], mode='sum')
        exp_value2 = Lambda(lambda x: K.dot(x, ones))(value2)
        q_value = merge([exp_value2, sum_adv], mode='sum')
    act = Input(shape=(num_actions,))
    q_value_act = merge([q_value, act], mode='dot')
    model = Model(input=[state, act], output=[q_value_act, q_value])
    return model


def get_output_folder(parent_dir, env_name):
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir += '-run{}'.format(experiment_id)
    subprocess.call(["mkdir", "-p", parent_dir])
    return parent_dir


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    parser.add_argument('--input_shape', nargs=2, type=int, default=None,
                        help='Input shape')
    parser.add_argument('--num_frame', default=1, type=int,
                        help='Number of frames in a state')
    parser.add_argument('--discount', default=0.99, type=float,
                        help='Discount factor gamma')
    parser.add_argument('--replay_buffer_size', default=100000, type=int,
                        help='Replay buffer size')
    parser.add_argument('--online_train_interval', default=4, type=int,
                        help='Interval to train the online network')
    parser.add_argument('--target_reset_interval', default=10000, type=int,
                        help='Interval to reset the target network')
    parser.add_argument('--num_burn_in', default=25000, type=int,
                        help='Number of samples filled in memory before update')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='How many samples in each minibatch')

    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Learning rate alpha')
    parser.add_argument('--explore_prob', default=0.05, type=float,
                        help='Exploration probability in epsilon-greedy')
    parser.add_argument('--num_train', default=5000000, type=int,
                        help='Number of training sampled interactions with the environment')
    parser.add_argument('--max_episode_length', default=999999, type=int,
                        help='Maximum length of an episode')
    parser.add_argument('--save_interval', default=100000, type=int,
                        help='Interval to save weights and memory')

    parser.add_argument('--model_name', default='dqn', type=str,
                        help='Model name')

    parser.add_argument('--eval_interval', default=10000, type=int,
                        help='Evaluation interval')
    parser.add_argument('--eval_episodes', default=20, type=int,
                        help='Number of episodes in evaluation')

    parser.add_argument('--double_q', default=False, type=bool,
                        help='Invoke double Q net')

    parser.add_argument('--do_render', default=False, type=bool,
                        help='Do rendering or not')

    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)
    args.output = get_output_folder(args.output, args.env)

    env = gym.make(args.env)
    num_actions = env.action_space.n
    opt_adam = Adam(lr=args.learning_rate)

    model_online = create_model(args.num_frame, args.input_shape,
        num_actions, model_name=args.model_name)
    model_target = create_model(args.num_frame, args.input_shape,
        num_actions, model_name=args.model_name)

    q_network = {'online': model_online, 'target': model_target}

    preproc = AtariPreprocessor(args.input_shape)
    memory = ReplayMemory(args.replay_buffer_size, args.num_frame)

    policy_random = UniformRandomPolicy(num_actions)
    policy_train = LinearDecayGreedyEpsilonPolicy(args.explore_prob + 0.5,
                                                  args.explore_prob,
                                                  args.num_train)
    policy_eval = GreedyEpsilonPolicy(args.explore_prob)
    policy = {'random': policy_random, 'train': policy_train, 'eval': policy_eval}

    agent = DQNAgent(num_actions, q_network, preproc, memory, policy, args)
    agent.compile([mean_huber_loss, null_loss], opt_adam)

    print '########## training #############'
    agent.fit(env)



if __name__ == '__main__':
    main()

