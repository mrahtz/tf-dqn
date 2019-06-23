import os
import time

import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBatch
from utils import tensor_index, huber_loss


class Model:

    def __init__(self, policy_fn, obs_shape, n_actions, seed, discount, lr, gradient_clip, double_dqn, save_dir=None):
        self.save_args(locals())
        self.n_actions = n_actions
        self.save_dir = save_dir

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.random.set_random_seed(seed)

            obs1_ph = tf.placeholder(tf.float32, (None,) + obs_shape, name='o1')
            obs2_ph = tf.placeholder(tf.float32, (None,) + obs_shape, name='o2')
            acts_ph = tf.placeholder(tf.int32, (None,), name='a1')
            rews_ph = tf.placeholder(tf.float32, (None,), name='r')
            done_ph = tf.placeholder(tf.float32, (None,), name='done')
            impw_ph = tf.placeholder(tf.float32, (None,), name='impw')

            with tf.variable_scope('main'):
                policy = policy_fn(obs_shape, n_actions)
                q1s_main = policy(obs1_ph)
                q2s_main = policy(obs2_ph)
                assert q1s_main.shape.as_list() == [None, n_actions]
                assert q2s_main.shape.as_list() == [None, n_actions]

            with tf.variable_scope('target'):
                policy = policy_fn(obs_shape, n_actions)
                q2s_target = policy(obs2_ph)
                assert q2s_target.shape.as_list() == [None, n_actions]

            params = tf.trainable_variables('main')
            params_target = tf.trainable_variables('target')
            target_update_ops = [param_target.assign(param) for param_target, param in zip(params_target, params)]

            pi = tf.argmax(q1s_main, axis=1)
            assert pi.shape.as_list() == [None]

            q1_main = tensor_index(q1s_main, acts_ph)
            assert q1_main.shape.as_list() == [None]
            if double_dqn:
                # Double DQN: we choose the action using the main network,
                # but evaluate it using the target network
                a = tf.argmax(q2s_main, axis=1)
                q_target = tensor_index(q2s_target, a)
            else:
                # Vanilla DQN: we both choose the action and evaluate it
                # using the target network
                q_target = tf.reduce_max(q2s_target, axis=1)

            backup = rews_ph + discount * (1 - done_ph) * q_target
            assert backup.shape.as_list() == [None]

            td_error = q1_main - tf.stop_gradient(backup)
            assert td_error.shape.as_list() == [None]

            assert impw_ph.shape.as_list() == [None]
            weighted_td_error = impw_ph * td_error
            assert weighted_td_error.shape.as_list() == [None]

            # From the paper:
            #
            #   We also found it helpful to clip the error term from the update
            #     r + gamma * max_a' Q(s', a') - Q(s, a)
            #   to be between -1 and 1. Because the absolute value loss function |x| has a derivative of -1 for all
            #   negative values of x and a derivative of 1 for all positive values of x, clipping the squared error to
            #   be between -1 and 1 corresponds to using an absolute value loss function for errors outside of the
            #   (-1, 1). interval. This form of error clipping further improved the stability of the algorithm.
            #
            # As https://openai.com/blog/openai-baselines-dqn/ notes, this is somewhat ambiguous. The key piece of
            # information is that the loss should be equivalent to an absolute loss outside (-1, 1). That implies
            # clipping the gradient rather than clipping the value (which would be a bad idea anyway, because it would
            # produce a flat derivative outside (-1, 1)).
            weighted_td_error_clipped = huber_loss(weighted_td_error, delta=1)
            assert weighted_td_error_clipped.shape.as_list() == [None]

            loss = tf.reduce_mean(weighted_td_error_clipped)
            assert loss.shape.as_list() == []

            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            grads_and_vars = optimizer.compute_gradients(loss, var_list=tf.trainable_variables('main'))
            grads_and_vars_clipped = [(tf.clip_by_norm(grad, gradient_clip), var) for grad, var in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars_clipped)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

        self.obs1_ph = obs1_ph
        self.acts_ph = acts_ph
        self.rews_ph = rews_ph
        self.obs2_ph = obs2_ph
        self.done_ph = done_ph
        self.impw_ph = impw_ph
        self.q1s_main = q1s_main
        self.q2s_target = q2s_target
        self.q1 = q1_main
        self.pi = pi
        self.loss = loss
        self.train_op = train_op
        self.sess = sess
        self.saver = saver
        self.target_update_ops = target_update_ops
        self.td_error = td_error

        self.update_target()

    def save_args(self, locals):
        locals = dict(locals)
        self.args = locals
        del self.args['self']
        self.args['save_dir'] = None

    def __getstate__(self):
        with self.graph.as_default():
            vars = tf.trainable_variables()
        names = [v.name for v in vars]
        values = self.sess.run(vars)
        params = {k: v for k, v in zip(names, values)}
        state = self.args, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        with self.graph.as_default():
            ops = []
            for var in tf.trainable_variables():
                assign = var.assign(params[var.name])
                ops.append(assign)
            self.sess.run(ops)

    def step(self, obs, random_action_prob):
        if np.random.rand() < random_action_prob:
            return np.random.choice(self.n_actions)
        else:
            return self.sess.run(self.pi, feed_dict={self.obs1_ph: [obs]})[0]

    def q(self, obs, act):
        return self.sess.run(self.q1,
                             feed_dict={self.obs1_ph: [obs],
                                        self.acts_ph: [act]})[0]

    def update_target(self):
        self.sess.run(self.target_update_ops)

    def train(self, batch: ReplayBatch):
        _, loss, td_errors = self.sess.run([self.train_op, self.loss, self.td_error],
                                           feed_dict={self.obs1_ph: batch.obs1,
                                                      self.acts_ph: batch.acts,
                                                      self.rews_ph: batch.rews,
                                                      self.obs2_ph: batch.obs2,
                                                      self.done_ph: batch.done,
                                                      self.impw_ph: batch.impw})
        return loss, td_errors

    def calculate_td(self, obs1, act, reward, obs2, done):
        return self.sess.run(self.td_error, feed_dict={self.obs1_ph: [obs1],
                                                       self.acts_ph: [act],
                                                       self.rews_ph: [reward],
                                                       self.obs2_ph: [obs2],
                                                       self.done_ph: [done]})[0]

    def save(self):
        save_id = int(time.time())
        self.saver.save(self.sess, os.path.join(self.save_dir, 'model'), save_id)

    def load(self, load_dir):
        ckpt = tf.train.latest_checkpoint(load_dir)
        self.saver.restore(self.sess, ckpt)
