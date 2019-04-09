import gym
import control
import itertools
import numpy as np
import matplotlib.pyplot as plt

from cartpole_continuous import ContinuousCartPoleEnv

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

        K, S, E = control.lqr(sys, Q, R)
        self.K = -K.reshape((4,))

    def predict(self, state):
        action = state.reshape((4,)).dot(self.K)
        return action


def run_episode(env, control, n_episodes, show_episode=False,
                alpha=0.5, discount=1.0, plot=False):

    starting_pos = np.zeros((n_episodes, 5))

    for i in range(n_episodes):
        tot_reward = 0

        state = env.reset(0.9)
        start_state = state
        for t in itertools.count():
            action = control.predict(state)
            next_state, reward, done, info = env.step(action)

            tot_reward += reward

            # Print and cleanup
            print("\rStep {} @ episode {}/{} ({})".format(t, i, n_episodes,
                                                          tot_reward), end="")

            if done or t == 200:
                starting_pos[i, 0:4] = start_state[0:4]
                starting_pos[i, 4] = t==200
                break
            state = next_state

        print()

    env.close()

    return starting_pos

env = ContinuousCartPoleEnv()

control = ControlPolicy(env)

A = run_episode(env, control, 100000)
