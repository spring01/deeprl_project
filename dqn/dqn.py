
import os
import numpy as np
import random


class DQNAgent(object):

    def __init__(self, num_actions, q_network, preproc, memory,
                 policy, args):
        self.num_actions = num_actions
        self.q_network = q_network
        self.preproc = preproc
        self.memory = memory
        self.policy = policy
        self.args = args
        self.null_act = np.zeros([1, self.num_actions])
        self.null_target = np.zeros([self.args.batch_size, self.num_actions])

    def compile(self, loss, optimizer):
        self.q_network['online'].compile(loss=loss, optimizer=optimizer)
        self.q_network['target'].compile(loss=loss, optimizer=optimizer)

    def fit(self, env):
        self.update_target()

        # filling in self.args.num_burn_in states
        print '########## burning in some samples #############'
        while self.memory.length() < self.args.num_burn_in:
            env.reset()
            state, state_mem, _, done = self.get_state(env, 0)

            for ep_len in xrange(self.args.max_episode_length):
                if done:
                    break
                q_online = self.predict_online(state)
                act = self.policy['train'].select_action(q_online, 0)
                state, state_mem_next, reward, done = self.get_state(env, act)

                # store transition into replay memory
                mem = state_mem, act, reward, state_mem_next, done
                self.memory.append(mem)
                state_mem = state_mem_next

        iter_num = 0
        while iter_num <= self.args.num_train:
            env.reset()
            state, state_mem, _, done = self.get_state(env, 0)

            print '########## begin new episode #############'
            for ep_len in xrange(self.args.max_episode_length):
                if done:
                    break

                # get online q value and get action
                q_online = self.predict_online(state)
                act = self.policy['train'].select_action(q_online, iter_num)

                # do action to get the next state
                state, state_mem_next, reward, done = self.get_state(env, act)
                reward = self.preproc.clip_reward(reward)

                # store transition into replay memory
                mem = state_mem, act, reward, state_mem_next, done
                self.memory.append(mem)

                # update networks
                if _every(iter_num, self.args.online_train_interval):
                    self.train_online()
                if _every(iter_num, self.args.target_reset_interval):
                    self.update_target()

                # evaluation
                if _every(iter_num, self.args.eval_interval):
                    print '########## evaluation #############'
                    self.evaluate(env)

                # save model
                if _every(iter_num, self.args.save_interval):
                    weights_save_name = os.path.join(self.args.output, 'online_{:d}.h5'.format(iter_num))
                    print '########## saving models and memory #############'
                    self.q_network['online'].save_weights(weights_save_name)
                    print 'online weights written to {:s}'.format(weights_save_name)
                    memory_save_name = os.path.join(self.args.output, 'memory.pickle')
                    self.memory.save(memory_save_name)
                    print 'replay memory written to {:s}'.format(memory_save_name)

                state_mem = state_mem_next
                iter_num += 1
                if _every(iter_num, 100):
                    self.print_loss()
            print '{:d} out of {:d} iterations'.format(iter_num, self.args.num_train)

    def predict_online(self, state):
        return self.q_network['online'].predict([state, self.null_act])[1]

    def train_online(self):
        input_b, act_b, target_b = self.get_batch(self.args.batch_size)
        self.q_network['online'].train_on_batch([input_b, act_b], [target_b, self.null_target])

    def print_loss(self):
        input_b, act_b, target_b = self.get_batch(self.args.batch_size)
        null_target = np.zeros(act_b.shape)
        loss_online = self.q_network['online'].evaluate([input_b, act_b],
            [target_b, null_target], verbose=0)
        loss_target = self.q_network['target'].evaluate([input_b, act_b],
            [target_b, null_target], verbose=0)
        print 'losses:', loss_online[0], loss_target[0]

    def evaluate(self, env):
        total_reward = 0.0
        for episode in xrange(self.args.eval_episodes):
            env.reset()
            state, state_mem, episode_reward, done = self.get_state(env, 0)

            for ep_len in xrange(self.args.max_episode_length):
                if done:
                    break

                # get online q value and get action
                q_online = self.predict_online(state)
                act = self.policy['eval'].select_action(q_online)

                # do action to get the next state
                state, state_mem_next, reward, done = self.get_state(env, act)
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
        for _ in xrange(self.args.num_frame):
            if not done:
                obs_next, obs_reward, obs_done, _ = env.step(action)
                if self.args.do_render:
                    env.render()
                obs_next_mem = self.preproc.obs_to_obs_mem(obs_next)
                reward += obs_reward
                done = done or obs_done
            state_mem_next.append(obs_next_mem)
        state_mem_next = np.stack(state_mem_next)
        state_next = self.preproc.state_mem_to_state(state_mem_next)
        state_next = np.stack([state_next])
        return state_next, state_mem_next, reward, done

    def update_target(self):
        print 'update update update update update'
        online_weights = self.q_network['online'].get_weights()
        self.q_network['target'].set_weights(online_weights)

    def get_batch(self, batch_size):
        mini_batch = self.memory.sample(batch_size)
        input_b = []
        act_b = []
        one_hot_eye = np.eye(self.num_actions, dtype=np.float32)
        input_b_n = []
        for st_m, act, rew, st_m_n, done_b in mini_batch:
            st = self.preproc.state_mem_to_state(st_m)
            input_b.append(st)
            act_b.append(one_hot_eye[act].copy())
            st_n = self.preproc.state_mem_to_state(st_m_n)
            input_b_n.append(st_n)
        input_b = np.stack(input_b)
        act_b = np.stack(act_b)
        input_b_n = np.stack(input_b_n)

        q_target_b_n = self.q_network['target'].predict([input_b_n, act_b])[1]
        target_b = []
        for q_target, (_, _, rew, _, done_b) in zip(q_target_b_n, mini_batch):
            full_reward = rew
            if not done_b:
                if self.args.double_q:
                    new_act = np.argmax(q_target)
                    long_term = q_target[new_act]
                else:
                    long_term = np.max(q_target)
                full_reward += self.args.discount * long_term
            target_b.append([full_reward])
        target_b = np.stack(target_b)
        return input_b, act_b, target_b

def _every(iter_num, interval):
    return not (iter_num % interval)
