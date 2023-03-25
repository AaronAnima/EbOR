import time
import pickle
import random
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import pybullet as p
import os
from ebor.Envs.base_env import BallEnv
from ebor.Envs.utils import *
from ebor.Envs.target_generation import get_example_positions

class BallGym(BallEnv):
    def __init__(self, pattern='clustering', **kwargs):
        super().__init__(**kwargs)
        self.max_action = kwargs['max_action']
        self.num_class = len(self.catetory_list)
        self.action_space = spaces.Box(-self.max_action, self.max_action, shape=(2*self.num_class*self.n_boxes_per_class, ), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(2*self.num_class*self.n_boxes_per_class, ), dtype=np.float32)
        self.cur_steps = 0
        self.pattern = pattern
        self._seed()

    def nop_action(self):
        action = np.zeros((self.n_boxes, 2))
        return action

    def nop(self):
        """
        no operation at this time step
        """
        action = self.nop_action()
        return self.step(action)

    def step(self, vels, step_size=4, centralized=False, soft_collision=False):
        """
        vels: [3*n_balls_per_class, 2], 2-d linear velocity
        for each dim,  [-max_vel,  max_vel]
        """
        # action: numpy, [num_box*2]
        collision_num = 0
        old_state = self.get_state(self.balls_list)
        old_pos = old_state.reshape(self.num_class*self.n_boxes_per_class, 2)

        vels = np.reshape(vels, (self.n_boxes, 2))
        max_vel_norm = np.max(np.abs(vels))
        scale_factor = self.max_action / (max_vel_norm+1e-7)
        scale_factor = np.min([scale_factor, 1])
        vels = scale_factor * vels
        max_vel = vels.max()
        if max_vel > self.max_action:
            print(f'!!!!!current max velocity {max_vel} exceeds max action {self.max_action}!!!!!')
        self.set_velocity(vels, self.balls_list)
        for _ in range(step_size):
            p.stepSimulation(physicsClientId=self.cid)
            collision_num += self.get_collision_num(self.balls_list, centralized)
        collision_num /= step_size 

        ''' M3D20 modify '''
        if not soft_collision:
            collision_num = (collision_num > 0) 

        r = 0
        # judge if is done
        self.cur_steps += 1
        is_done = self.cur_steps >= self.max_episode_len

        new_pos = self.get_state(self.balls_list).reshape(self.num_class*self.n_boxes_per_class, 2)
        delta_pos = new_pos - old_pos
        vel_err = np.max(np.abs(delta_pos*self.time_freq*self.bound/step_size - vels))/self.max_action
        vel_err_mean = np.mean(np.abs(delta_pos*self.time_freq*self.bound/step_size - vels))/self.max_action
        infos = {'delta_pos': delta_pos, 'collision_num': collision_num,
                'vel_err': vel_err, 'vel_err_mean': vel_err_mean,
                'is_done': is_done, 'progress': self.cur_steps / self.max_episode_len,
                'init_state': self.init_state, 'cur_steps': self.cur_steps, 'max_episode_len': self.max_episode_len}

        return self.get_state(self.balls_list), r, is_done, infos

    def reset(self, is_random=True, random_flip=True, random_rotate=True, remove_collision=True):
        self.num_episodes += 1
        t_s = time.time()
        if is_random:
            balls_dict = get_example_positions(self.n_boxes_per_class, self.catetory_list, 'random', self.bound, self.r)
        else:  # init to target distribution
            balls_dict = get_example_positions(self.n_boxes_per_class, self.catetory_list, self.pattern, self.bound, self.r)

        for color, positions in balls_dict.items():
            self.balls[color] = self.add_balls(positions, color)

        self.balls_list = [value for sublist in self.balls.values() for value in sublist]
        p.stepSimulation(physicsClientId=self.cid)
        if remove_collision:
            total_steps = 0
            while self.get_collision_num(self.balls_list) > 0:
                for _ in range(20):
                    p.stepSimulation(physicsClientId=self.cid)
                total_steps += 20
                if total_steps > 10000:
                    print('Warning! Reset takes too much trial!')
                    break
        
        self.cur_steps = 0
        self.init_state = self.get_state(self.balls_list)
        return self.init_state

    def sample_action(self):
        """
        sample a random action according to current action space
        return range [-self.max_action, self.max_action]
        """
        return np.random.normal(size=3*self.n_boxes_per_class * 2).clip(-1, 1) * self.max_action

    def get_obs(self):
        return self.sim.get_state()
    
    def _seed(self, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def seed(seed):
        np.random.seed(seed)
        random.seed(seed)

    def get_obs(self):
        return self.get_state(self.balls_list)
