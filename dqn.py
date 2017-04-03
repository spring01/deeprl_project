
import numpy as np
import random
import os
import cPickle as pickle


class DQNAgent(object):

    def __init__(self, num_actions, q_network, preproc, memory,
                 policy, args):
        self.num_actions = num_actions
        self.q_network = q_network
        self.preproc = preproc
        self.memory = memory
        self.policy = policy
        self.args = args

    def compile(self, loss, optimizer):
        self.q_network['online'].compile(loss=loss, optimizer=optimizer)
        self.q_network['target'].compile(loss=loss, optimizer=optimizer)

    def train_online(self):
        mini_batch = random.sample(self.memory, self.args.batch_size)
        input_b = []
        input_b_n = []
        for st_m, act, rew, st_m_n, done_b in mini_batch:
            st = self.preproc.state_mem_to_state(st_m)
            input_b.append(st)
            st_n = self.preproc.state_mem_to_state(st_m_n)
            input_b_n.append(st_n)
        input_b = np.stack(input_b)
        input_b_n = np.stack(input_b_n)

        q_target_b_n = self.q_network['target'].predict(input_b_n)
        target_b = self.q_network['online'].predict(input_b)
        for i, (_, act, rew, _, done_b) in enumerate(mini_batch):
            target_b[i, act] = rew
            if not done_b:
                if self.args.double_q:
                    new_act = np.argmax(q_target_b_n[i])
                    long_term = q_target_b_n[i, new_act]
                else:
                    long_term = np.max(q_target_b_n[i])
                target_b[i, act] += self.args.discount * long_term

        self.q_network['online'].train_on_batch(input_b, target_b)

    def fit(self, env):
        self.update_target()

        # filling in self.args.num_burn_in states
        print '########## burning in some samples #############'
        while len(self.memory) < self.args.num_burn_in:
            env.reset()
            state_mem, _, done = self.get_state(env, 0)

            for ep_len in xrange(self.args.max_episode_length):
                if done:
                    break
                act = self.policy['random'].select_action()
                state_mem_next, reward, done = self.get_state(env, act)

                # store transition into replay memory
                mem = state_mem, act, reward, state_mem_next, done
                self.memory.append(mem)
                state_mem = state_mem_next

        iter_num = 0
        while iter_num <= self.args.num_train:
            env.reset()
            state_mem, _, done = self.get_state(env, 0)

            print '########## begin new episode #############'
            for ep_len in xrange(self.args.max_episode_length):
                if done:
                    break
                # get online q value and get action
                state = self.preproc.state_mem_to_state(state_mem)
                input_state = np.stack([state])
                q_online = self.q_network['online'].predict(input_state)
                act = self.policy['train'].select_action(q_online, iter_num)

                # do action to get the next state
                state_mem_next, reward, done = self.get_state(env, act)
                reward = self.preproc.clip_reward(reward)

                # store transition into replay memory
                mem = (state_mem, act, reward, state_mem_next, done)
                self.memory.append(mem)

                # update networks
                self.train_online()
                if not (iter_num % self.args.target_reset_interval):
                    self.update_target()

                if not (iter_num % self.args.eval_interval):
                    print '########## evaluation #############'
                    self.evaluate(env)

                state_mem = state_mem_next
                iter_num += 1
            print '{:d} out of {:d} iterations'.format(iter_num, self.args.num_train)

    def evaluate(self, env):
        total_reward = 0.0
        for episode in range(self.args.eval_episodes):
            env.reset()
            state_mem, episode_reward, done = self.get_state(env, 0)

            # episode loop
            for ep_len in xrange(self.args.max_episode_length):
                if done:
                    break
                state = self.preproc.state_mem_to_state(state_mem)
                # get online q value and get action
                input_state = np.stack([state])
                q_online = self.q_network['online'].predict(input_state)
                act = self.policy['eval'].select_action(q_online)

                # do action to get the next state
                state_mem_next, reward, done = self.get_state(env, act)
                episode_reward += reward
                state_mem = state_mem_next

            print '  episode reward: {:f}'.format(episode_reward)
            total_reward += episode_reward
            avg_reward = total_reward / self.args.eval_episodes
        print 'average episode reward: {:f}'.format(avg_reward)

    def get_state(self, env, action):
        state_mem_next = []
        reward = 0.0
        done = False
        for _ in range(self.args.num_frame):
            if not done:
                obs_next, obs_reward, obs_done, _ = env.step(action)
                if self.args.do_render:
                    env.render()
                obs_next_mem = self.preproc.obs_to_obs_mem(obs_next)
                reward += obs_reward
                done = done or obs_done
            state_mem_next.append(obs_next_mem)
        state_mem_next = np.stack(state_mem_next)
        return state_mem_next, reward, done

    def update_target(self):
        print 'update update update update update'
        online_weights = self.q_network['online'].get_weights()
        self.q_network['target'].set_weights(online_weights)

