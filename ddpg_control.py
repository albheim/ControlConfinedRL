import numpy as np
import tensorflow as tf
import gym
import itertools
import matplotlib.pyplot as plt

import core
from core import get_vars
from cartpole_continuous_double import ContinuousCartPoleEnv
from lqr import ControlPolicy

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
         control_policy=ControlPolicy,
         n_episodes=10000, replay_size=int(1e6), gamma=0.99, show_steps=50,
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
         act_noise=0.1, max_ep_len=200, logger_kwargs=dict(), save_freq=1):

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ctrl_pol = control_policy(env)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q, q_pi = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Target networks
    with tf.variable_scope('target'):
        # Note that the action placeholder going to actor_critic here is
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ, _, q_pi_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q-backup)**2)

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=5):
        tot_len, tot_ret = 0, 0
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            o_ctrl, cost_ctrl = np.array(o), 0
            while not (d or (ep_len == 5*max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                test_env.render()
                a_ctrl = np.array([ctrl_pol.predict(o_ctrl)])
                o_ctrl, _, _, info = test_env.step(a_ctrl, 0)
                cost_ctrl += info["cost"]
                a = get_action(o, 0)
                o, r, d, info = test_env.step(a, 1)
                ep_ret += (r - info["cost"])
                ep_len += 1
            tot_len += ep_len
            tot_ret += ep_ret
        test_env.close()
        print("\n avg reward {:.5} and episode length {} over {} trials, cost/step rl/lqr {:.5}/{:.5}".
              format(tot_ret/n, tot_len/n, n, (tot_len - tot_ret) / tot_len, cost_ctrl / tot_len))

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    o_ctrl = np.array(o)#env.state[0]

    for t in range(start_steps):
        #a = env.action_space.sample()
        a = np.array([ctrl_pol.predict(o)])
        o2, r, d, info = env.step(a, 1)
        r -= info["cost"]
        replay_buffer.store(o, a, r, o2, d)
        o = o2
        if d:
            o = env.reset()

    fails = 0
    takeover = False
    cost = 0
    cost_ctrl = 0
    retrain_steps = 0

    # Setup plotting
    plt.ion()
    fig, ax = plt.subplots()
    plot = ax.scatter([], [])
    plot_ctrl = ax.scatter([], [])
    plot_act = ax.scatter([], [])
    plot_ctrl_act = ax.scatter([], [])
    ax.legend(["ddpg cost", "lqr cost", "ddpg a", "lqr a"])
    ax.set_xlabel("time")
    ax.set_ylabel("cost and action")
    ax.set_ylim(0, 3)

    # Main loop: collect experience in env and update/log each epoch
    for t in itertools.count():
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy (with some noise, via act_noise).
        """
        env.render(takeover=takeover)

        # Step lqr
        a_ctrl = np.array([ctrl_pol.predict(o_ctrl)])
        o_ctrl, _, _, info = env.step(a_ctrl, 0)
        cost_ctrl += info["cost"]

        # Step ddpg
        scaler = 1 / (1 + np.exp(start_steps - t))
        takeover = np.abs(o[2]) > 0.25 * scaler or np.abs(o[0]) > 1 * scaler
        if takeover:
            a = np.array([ctrl_pol.predict(o)])
        else:
            a = get_action(o, act_noise)
        o2, r, d, info = env.step(a, 1)

        cost += info["cost"]
        r -= info["cost"]
        ep_ret += r
        ep_len += 1
        retrain_steps += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if t==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        print("\rSteps {:5}, fails {:3}, cost/step {:7.2}, ep_len {:5}, disturbance {:6.3}   "
              .format(t, fails, cost/(t % max_ep_len), ep_len, info["disturbance"] if info["push"] else 0.0), end="")

        if np.random.rand() * max_ep_len < 1:
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(max_ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']}

                # Q-learning update
                outs = sess.run([q_loss, q, train_q_op], feed_dict)

                # Policy update
                outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)

            cost /= retrain_steps
            cost_ctrl /= retrain_steps

            point = np.array([0.02 * (t + start_steps), cost]).reshape((1, 2))
            array = plot.get_offsets()
            array = np.append(array, point, axis=0)
            plot.set_offsets(array)

            point = np.array([0.02 * (t + start_steps), cost_ctrl]).reshape((1, 2))
            array = plot_ctrl.get_offsets()
            array = np.append(array, point, axis=0)
            plot_ctrl.set_offsets(array)

            point = np.array([0.02 * (t + start_steps), np.abs(a_ctrl)]).reshape((1, 2))
            array = plot_act.get_offsets()
            array = np.append(array, point, axis=0)
            plot_act.set_offsets(array)

            point = np.array([0.02 * (t + start_steps), np.abs(a)]).reshape((1, 2))
            array = plot_ctrl_act.get_offsets()
            array = np.append(array, point, axis=0)
            plot_ctrl_act.set_offsets(array)

            ax.set_xlim(0.02 * start_steps, 10 * np.ceil(0.002 * (t + start_steps + 1)))
            fig.canvas.draw()
            plt.pause(0.005)

            cost = 0
            cost_ctrl = 0
            retrain_steps = 0

            env.state[0] = np.array(env.state[1])
            o_ctrl = env.state[0]
        if d:
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            o_ctrl = np.array(o)
            fails += 1
            print()

    # End of epoch wrap-up
        if t > 0 and t % (show_steps * max_ep_len) == 0:
            # Test the performance of the deterministic version of the agent.
            test_agent()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    ddpg(lambda : ContinuousCartPoleEnv(), actor_critic=core.mlp_actor_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         gamma=args.gamma, seed=args.seed)
