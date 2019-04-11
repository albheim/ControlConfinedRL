"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class ContinuousCartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 30 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(np.array([-1]), np.array([1]))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = [None, None]

        self.steps_beyond_done = None

        scaler = 100.0
        self.Q = np.diag([10, 1, 10, 1]) / scaler
        self.R = 10 / scaler

        self.disturbance_prob = 0.01
        self.disturbance_max_time = 0.4
        self.disturbance_count = 0
        self.disturbance_max_val = 1
        self.disturbance = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, idx):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        action = action[0]
        state = self.state[idx]
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * action
        push = False
        if self.disturbance_count > 0:
            if idx == 1:
                self.disturbance_count -= 1
            force += self.disturbance
            push = True
        elif idx == 1 and self.np_random.rand() < self.disturbance_prob:
            self.disturbance_count = self.disturbance_max_time / self.tau
            self.disturbance = self.disturbance_max_val * 2 * (0.5 - self.np_random.rand())
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state[idx] = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        state = np.array(self.state[idx])

        cost = state.dot(self.Q).dot(state) + self.R * action**2

        reward = 0.0
        if idx == 1:
            if not done:
                reward = 1.0
            elif self.steps_beyond_done is None:
                # pole just fell!
                self.steps_beyond_done = 0
                reward = 1.0
            elif idx == 1:
                if self.steps_beyond_done == 0:
                    logger.warn("you are calling 'step()' even though this environment has already returned done = true. you should always call 'reset()' once you receive 'done = true' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1

        return state, reward, done, {"push": push,
                                    "disturbance": self.disturbance,
                                    "cost": cost}

    def reset(self, d=0.5):
        self.state[0] = self.np_random.uniform(low=-d, high=d, size=(4,))
        self.state[1] = np.array(self.state[0])
        self.steps_beyond_done = None
        return np.array(self.state[1])

    def render(self, mode='human', takeover=False):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            self.cart = [rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) for _ in range(2)]
            self.carttrans = [rendering.Transform() for _ in range(2)]
            for c, t in zip(self.cart, self.carttrans):
                c.add_attr(t)
                self.viewer.add_geom(c)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = [rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) for _ in range(2)]
            self.poletrans = [rendering.Transform(translation=(0, axleoffset)) for _ in range(2)]
            for i in range(2):
                pole[i].set_color(.8,.6,.4)
                pole[i].add_attr(self.poletrans[i])
                pole[i].add_attr(self.carttrans[i])
                self.viewer.add_geom(pole[i])
            self.axle = [rendering.make_circle(polewidth/2) for _ in range(2)]
            for i in range(2):
                self.axle[i].add_attr(self.poletrans[i])
                self.axle[i].add_attr(self.carttrans[i])
                self.axle[i].set_color(.5,.5,.8)
                self.viewer.add_geom(self.axle[i])
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        for i in range(2):
            pole[i].v = [(l,b), (l,t), (r,t), (r,b)]

            x = self.state[i]
            cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
            self.carttrans[i].set_translation(cartx, carty)
            self.poletrans[i].set_rotation(-x[2])

        if takeover:
            self.cart[1].set_color(200, 0, 0)
        else:
            self.cart[1].set_color(0, 200, 0)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
