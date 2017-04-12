
import os
import numpy as np


class DRQNAgent(object):

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
        while len(self.memory) < self.args.num_burn_in:
            env.reset()
            state, state_mem, _, done = self.get_state(env, 0)

            act = self.policy['random'].select_action()
            for ep_len in xrange(self.args.max_episode_length):
                if done:
                    break
                if _every(ep_len, self.args.action_change_interval):
                    act = self.policy['random'].select_action()
                state, state_mem_next, reward, done = self.get_state(env, act)

                # store transition into replay memory
                mem = state_mem, act, reward, state_mem_next, done
                self.memory.append(mem)
                state_mem = state_mem_next

        iter_num = 0
        eval_flag = False
        while iter_num <= self.args.num_train:
            env.reset()
            state, state_mem, _, done = self.get_state(env, 0)

            print '########## begin new episode #############'
            q_online = self.predict_online(state)
            act = self.policy['train'].select_action(q_online, iter_num)
            for ep_len in xrange(self.args.max_episode_length):
                if done:
                    break

                # get online q value and get action
                if _every(ep_len, self.args.action_change_interval):
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
                    if self.args.double_q:
                        self.train_online_double()
                    else:
                        self.train_online()
                if _every(iter_num, self.args.target_reset_interval):
                    self.update_target()

                # set evaluation flag
                if _every(iter_num, self.args.eval_interval):
                    eval_flag = True

                # save model
                if _every(iter_num, self.args.save_interval):
                    weights_save_name = os.path.join(self.args.output,
                        'online_{:d}.h5'.format(iter_num))
                    print '########## saving models and memory #############'
                    self.q_network['online'].save_weights(weights_save_name)
                    print 'online weights written to {:s}'.format(weights_save_name)
                    memory_save_name = os.path.join(self.args.output, 'memory.pickle')
                    self.memory.save(memory_save_name)
                    print 'replay memory written to {:s}'.format(memory_save_name)

                state_mem = state_mem_next
                iter_num += 1
                if _every(iter_num, self.args.print_loss_interval):
                    self.print_loss()
            # evaluation
            if eval_flag:
                eval_flag = False
                print '########## evaluation #############'
                self.evaluate(env)
            print '{:d} out of {:d} iterations'.format(iter_num, self.args.num_train)

    def predict_online(self, state):
        return self.q_network['online'].predict([state, self.null_act])[1]

    def train_online(self):
        mini_batch, input_b, act_b, input_b_n = self.get_batch()
        target_b = self.get_target(mini_batch, input_b_n, act_b)
        self.q_network['online'].train_on_batch([input_b, act_b],
            [target_b, self.null_target])

    def train_online_double(self):
        mini_batch, input_b, act_b, input_b_n = self.get_batch()
        online_net, target_net = self.roll_online_target()
        target_b = self.get_target_double(mini_batch, input_b_n, act_b,
            online_net, target_net)
        online_net.train_on_batch([input_b, act_b],
            [target_b, self.null_target])

    def print_loss(self):
        mini_batch, input_b, act_b, input_b_n = self.get_batch()
        if self.args.double_q:
            online_net, target_net = self.roll_online_target()
            target_b = self.get_target_double(mini_batch, input_b_n, act_b,
                online_net, target_net)
        else:
            target_b = self.get_target(mini_batch, input_b_n, act_b)
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
            state_mem_next.append(np.stack([obs_next_mem], axis=2))
        state_mem_next = np.stack(state_mem_next)
        state_next = self.preproc.state_mem_to_state(state_mem_next)
        state_next = np.stack([state_next])
        return state_next, state_mem_next, reward, done

    def update_target(self):
        print 'update update update update update'
        online_weights = self.q_network['online'].get_weights()
        self.q_network['target'].set_weights(online_weights)

    def get_batch(self):
        mini_batch = self.memory.sample(self.args.batch_size)
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
        return mini_batch, input_b, act_b, input_b_n

    def get_target(self, mini_batch, input_b_n, act_b):
        q_target_b_n = self.q_network['target'].predict([input_b_n, act_b])[1]
        target_b = []
        for qtn, (_, _, rew, _, db) in zip(q_target_b_n, mini_batch):
            full_reward = rew
            if not db:
                full_reward += self.args.discount * np.max(qtn)
            target_b.append([full_reward])
        return np.stack(target_b)

    def roll_online_target(self):
        if np.random.rand() < 0.5:
            online_net = self.q_network['target']
            target_net = self.q_network['online']
        else:
            online_net = self.q_network['online']
            target_net = self.q_network['target']
        return online_net, target_net

    def get_target_double(self, mini_batch, input_b_n, act_b,
                          online_net, target_net):
        q_online_b_n = online_net.predict([input_b_n, act_b])[1]
        q_target_b_n = target_net.predict([input_b_n, act_b])[1]
        target_b = []
        ziplist = zip(q_online_b_n, q_target_b_n, mini_batch)
        for qon, qtn, (_, _, rew, _, db) in ziplist:
            full_reward = rew
            if not db:
                full_reward += self.args.discount * qon[np.argmax(qtn)]
            target_b.append([full_reward])
        return np.stack(target_b)

def _every(iter_num, interval):
    return not (iter_num % interval)
