import gym
import control
import itertools
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model

from cartpole_continuous import ContinuousCartPoleEnv


class MLPEstimator:
    def __init__(self):
        pass

    def predict(self, state, action):
        pass

    def update(self, state, action, td_target): pass

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
        Rlow = 10
        Rhigh = 15

        K, S, E = control.lqr(sys, Q, Rlow)
        self.Klow = -K.reshape((4,))
        K, S, E = control.lqr(sys, Q, Rhigh)
        self.Khigh = -K.reshape((4,))

        self.clip_counter = 0

    def predict(self, state, suggested_action=-1000):
        # clip suggestion by self.prediction +- deviation and approx deviation
        # in some growing way
        # or clip by self.predlow self.predhigh
        state = state.reshape((4,))
        action_low = state.dot(self.Klow)
        action_high = state.dot(self.Khigh)
        if action_low > suggested_action:
            suggested_action = action_low
        elif action_high < suggested_action:
            suggested_action = action_high
        else:
            self.clip_counter += 1

        return np.clip(suggested_action, -1, 1)


def run_episode(env, show_episode=0):
    tot_reward = 0

    model = MLPEstimator()
    policy = ControlPolicy(env)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.ylabel("offest")
    ax.xlabel("time")

    state = env.reset()
    for t in itertools.count():
        env.render()
        ax.plot(0.02 * t, abs(state[0]))
        fig.canvas.draw()

        action = policy.predict(state)
        next_state, reward, done, info = env.step(np.array([action]))

        tot_reward += reward

        print("\rStep {}, reward {}   ".format(t, tot_reward))

        if done:
            break
        state = next_state

    env.close()

    return policy, model


env = ContinuousCartPoleEnv(chaos=1)

policy, model = run_episode(env, show_episode=1)
