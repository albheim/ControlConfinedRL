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
        action = np.array([action])
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def update_action(self, state, target_action, sess=None, weight=1.0):
        sess = sess or tf.get_default_session()
        state = format_state(state)
        action = np.array([target_action])
        target = np.array([weight])
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

class ControlPolicy:
    def __init__(self, env):
        # Maybe have strict in start but slightly grow allowed prediction by
        # increasing deviation with successful suggestions by the estimator,
        # i.e. suggestions that were not clipped
        m = env.masspole
        M = env.masscart
        g = env.gravity
        l = env.length
        dt = env.tau
        tmp = 1/(13 * M + m)
        A = np.array([[0, 1, 0, 0],
                      [0, 0, -12*m*g*tmp, 0],
                      [0, 0, 0, 1],
                      [0, 0, 12*g*(m+M)*tmp/l, 0]])
        B = np.array([0, 13 * tmp, 0, -12*tmp/l]).reshape((4, 1))
        C = np.eye(4)
        D = np.zeros((4, 1))
        sys = control.ss(A, B, C, D, dt)

        Q = np.array([[10, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 10, 0],
                      [0, 0, 0, 1]])
        R = 10

        K, S, E = control.lqr(sys, Q, Rlow)
        self.K = -K.reshape((4,))

        self.clip_low = env.action_space.low[0]
        self.clip_high = env.action_space.high[0]

    def predict(self, state, suggested_action):
        # clip suggestion by self.prediction +- deviation and approx deviation
        # in some growing way
        # or clip by self.predlow self.predhigh
        action = state.dot(self.K)

        if action < 0.9 * self.clip_high and action > 0.9 * self.clip_low:
            action = suggested_action

        return np.clip(action, self.clip_low, self.clip_high)


def run_episode(env, policy, value, controlpolicy, n_episodes, show_episode=False, discount=1.0, plot=False):
    tot_reward = 0

    scores = np.zeros(10, 2)
    idx = 0

    if plot:
        plt.ion()
        fig, ax = plt.subplots()

        x_off = ax.scatter([], [])
        theta_off = ax.scatter([], [])
        model_action = ax.scatter([], [])
        lower = ax.scatter([], [])
        upper = ax.scatter([], [])
        empty = x_off.get_offsets()

        ax.legend(["x offset", "theta offset", "suggested action",
                   "lower bound", "upper bound"])

        ax.set_xlabel("time")
        ax.set_ylabel("offset")
        ax.set_ylim(-3, 3)

    for i in range(n_episodes):
        if plot:
            x_off.set_offsets(empty)
            theta_off.set_offsets(empty)
            model_action.set_offsets(empty)
            lower.set_offsets(empty)
            upper.set_offsets(empty)

        state = env.reset()
        for t in range(200):
            # Render
            if show_episode:
                env.render()

            # Step
            action_suggestion = policy.predict(state)[0]
            if controlpolicy is None:
                action = action_suggestion
            else:
                action = controlpolicy.predict(state, action_suggestion)
            next_state, reward, done, info = env.step(np.array([action]))

            tot_reward += reward

            # Plot
            if plot:
                point = np.array([0.02 * t, state[0]]).reshape((1, 2))
                array = x_off.get_offsets()
                array = np.append(array, point, axis=0)
                x_off.set_offsets(array)

                point = np.array([0.02 * t, state[2]]).reshape((1, 2))
                array = theta_off.get_offsets()
                array = np.append(array, point, axis=0)
                theta_off.set_offsets(array)

                point = np.array([0.02 * t, action_suggestion]).reshape((1, 2))
                array = model_action.get_offsets()
                array = np.append(array, point, axis=0)
                model_action.set_offsets(array)

                lower_action, upper_action = controlpolicy.get_bounds(state)
                point = np.array([0.02 * t, lower_action]).reshape((1, 2))
                array = lower.get_offsets()
                array = np.append(array, point, axis=0)
                lower.set_offsets(array)

                point = np.array([0.02 * t, upper_action]).reshape((1, 2))
                array = upper.get_offsets()
                array = np.append(array, point, axis=0)
                upper.set_offsets(array)

                ax.set_xlim(0, 10 * np.ceil(0.002 * (t + 1)))
                fig.canvas.draw()
                plt.pause(0.005)

            # Update MLPs
            td_target = reward + discount * value.predict(next_state)
            td_error = td_target - value.predict(state)

            if action == action_suggestion:
                ploss = policy.update(state, action_suggestion, td_error)
            else:
                ploss = policy.update_action(state, action)
            vloss = value.update(state, td_target)

            # Print and cleanup
            print("\rStep {}, policy loss \t{}\tvalue loss {} ".format(t, ploss, vloss), end="")

            if done:
                break
            state = next_state

    env.close()
    print()

    return controlpolicy, policy, value


env = ContinuousCartPoleEnv()

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

policy = PolicyEstimator()
value = ValueEstimator()
controlpolicy = ControlPolicy(env)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    run_episode(env, policy, value, controlpolicy, 10, show_episode=True, plot=True)
