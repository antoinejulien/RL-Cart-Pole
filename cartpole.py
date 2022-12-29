"""
Classic cart-pole system implemented by Rich Sutton et al.
Inspired highly from http://incompleteideas.net/sutton/book/code/pole.c
nearly copied from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
permalink: https://perma.cc/C9ZM-652R
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
import pygame
from pygame import gfxdraw

"""
 * @author Guillaume Gagné-Labelle
 * @student# 20174375
 * @date Dec 23, 2022
 * @project CartPole Problem - Final Project - PHY3075 - UdeM
"""

class CartPoleEnv():

    def __init__(self, render_mode="human", delete_limits=False):
        self.gravity = 9.80665  # m/s**2
        self.masscart = 1.  # kg
        self.masspole = 0.1  # kg
        self.total_mass = self.masspole + self.masscart  # kg
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length  # kg*m
        self.force_mag = 10.  # N
        self.deltaT = 0.02  # s; seconds between state updates
        self.metadata = {"render_modes": ["human", "quick_human", "rgb_array"], "render_fps": int(1/self.deltaT)}
        self.render_mode = render_mode
        self.delete_limits = delete_limits

        # Angle at which to fail the episode
        self.theta_threshold_radians = 30 * 2 * math.pi / 360  # rad
        self.x_threshold = 2.4  # m

        # min and max of each component of the observation vector
        self.low_state = [-1.5 * self.x_threshold, -1.5, -1.5*self.theta_threshold_radians, -math.radians(50)]
        self.high_state = [1.5 * self.x_threshold, 1.5, 1.5*self.theta_threshold_radians, math.radians(50)]

        self.screen_width = 1200
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        if action==1: force = self.force_mag
        elif action==0: force = -self.force_mag
        elif action==-1: force = 0

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x_dot = x_dot + self.deltaT * xacc
        x = x + self.deltaT * x_dot
        theta_dot = theta_dot + self.deltaT * thetaacc
        theta = theta + self.deltaT * theta_dot

        if self.delete_limits:
            if abs(x) > self.x_threshold: x = -x
            if theta > math.pi * 2: theta -= math.pi * 2
            if theta < - math.pi * 2: theta += math.pi * 2

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(abs(theta) > self.theta_threshold_radians or abs(x) > self.x_threshold)  # The pole is outside the limits
        if self.delete_limits:
            terminated = False

        if not terminated: reward = 1.0
        else: reward = 0.0

        if self.render_mode == "human" or self.render_mode == "quick_human": self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        if self.render_mode == "human" or self.render_mode == "quick_human": self.render()
        return np.array(self.state, dtype=np.float32)

    def energy(self):
        x_c, v_c, theta, w = self.state[:]

        l = self.length
        M = self.total_mass

        x_cm = (self.masscart * x_c + self.masspole * (x_c + l * math.sin(theta))) / M
        y_cm = self.masspole * l * math.cos(theta) / M
        d = math.sqrt((x_cm - x_c) ** 2 + y_cm ** 2)  # distance between the cart and the center of mass

        vx_cm = (self.masscart * v_c + self.masspole * (v_c + l * math.cos(theta) * w)) / M
        vy_cm = - self.masspole * l * math.sin(theta) * w / M
        v_cm = math.sqrt(vx_cm ** 2 + vy_cm ** 2)

        U = M * self.gravity * y_cm  # Potential energy
        K = M * v_cm ** 2 / 2  # Kinetic energy
        I = self.masscart * d ** 2 + self.masspole / (6*l) * (d**3 + (2*l -d)**3)  # moment of inertia of the system
        T = I * w ** 2 / 2  # Rotational energy of the system

        return U+T+K, U, K, T


    def render(self):

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human" or self.render_mode == "quick_human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            elif self.render_mode == "rgb_array":  # much faster
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        observation = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = (observation[0] * scale + self.screen_width / 2.0)  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-observation[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(self.surf, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))
        gfxdraw.filled_circle(self.surf, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()

        elif self.render_mode == "quick_human":
            pygame.event.pump()
            self.clock.tick()
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
