import gym
import control
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from cartpole_continuous import ContinuousCartPoleEnv


def format_state(state):
    return state.reshape((4,))

class PolicyEstimator():
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=[4], name="state")
            self.target = tf.placeholder(tf.float32, name="target")

            self.middle_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=10,
                activation_fn=tf.math.tanh,
                weights_initializer=tf.zeros_initializer)

            self.mu = tf.contrib.layers.fully_connected(
                inputs=self.middle_layer,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.mu = tf.squeeze(self.mu)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=self.middle_layer,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5

            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist._sample_n(1)
            self.action = tf.clip_by_value(self.action, env.action_space.low[0],
                                           env.action_space.high[0])

            self.loss = -self.normal_dist.log_prob(self.action) * self.target
            self.loss -= 1e-1 * self.normal_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = format_state(state)
        return sess.run(self.action, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = format_state(state)
        action = np.array([action]).reshape((1,))
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    def __init__(self, learning_rate=0.01, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=[4], name="state")
            self.target = tf.placeholder(tf.float32, name="target")

            self.middle_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=10,
                activation_fn=tf.math.tanh,
                weights_initializer=tf.zeros_initializer)

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.middle_layer,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = format_state(state)
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = format_state(state)
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def run_episode(env, policy, value, n_episodes, show_episode=False, discount=1.0):
    tot_reward = 0

    for i in range(n_episodes):

        state = env.reset()
        for t in range(200):
            # Render
            if show_episode and i % show_episode == 0:
                env.render()

            # Step
            action = policy.predict(state)[0]
            next_state, reward, done, info = env.step(np.array([action]))

            tot_reward += reward

            # Update MLPs
            td_target = reward + discount * value.predict(next_state)
            td_error = td_target - value.predict(state)

            ploss = policy.update(state, action, td_error)
            vloss = value.update(state, td_target)

            # Print and cleanup
            print("\rEpisode {:5} @ step {:3}, policy loss {:9.3} value loss {:9.3}".format(i, t, ploss[0], vloss[0]), end="")

            if done:
                break
            state = next_state

    env.close()
    print()

    return policy, value


env = ContinuousCartPoleEnv()

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

policy = PolicyEstimator()
value = ValueEstimator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    run_episode(env, policy, value, 100000, show_episode=500)
