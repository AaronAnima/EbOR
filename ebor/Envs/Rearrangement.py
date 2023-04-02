import random
from gym.utils import seeding
from gym.spaces import Dict, Discrete, Box
import numpy as np
import pybullet as p
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from ipdb import set_trace
from collections import OrderedDict
from ebor.Envs.base_env import BallEnv
from ebor.Envs.utils import *
from ebor.Envs.target_generation import sample_example_positions
from ebor.Envs.pseudo_likelihoods import get_pseudo_likelihood

class BallGym(BallEnv):
    def __init__(self, pattern='clustering', **kwargs):
        super().__init__(**kwargs)
        self.max_action = kwargs['max_action']
        self.num_classes = len(self.catetory_list)
        # create action space
        self.action_space = OrderedDict()
        for idx in range(self.num_objs):
            self.action_space[f'obj{idx}'] = Dict({"linear_vel": Box(-self.max_action, self.max_action, shape=(2,), dtype=np.float32)})
        self.action_space = Dict(self.action_space)
        # create observation space
        self.observation_space = OrderedDict()
        for idx in range(self.num_objs):
            self.observation_space[f'obj{idx}'] = Dict({"position": Box(-1, 1, shape=(2,), dtype=np.float32), "category": Discrete(self.num_classes)})
        self.observation_space = Dict(self.observation_space)
        self.cur_steps = 0
        self.exp_data = None
        self.num_episodes = 0
        self.init_state = None
        self.prev_state = None
        self.pattern = pattern
        self._seed()
    
    def flatten_states(self, states_list):
        """
        input: [state1, state2, ...]
        output: nparr, shape [num_states, (2+1)*num_objs]
        """
        states_np = []
        for state in states_list:
            state_np = []
            for obj_dict in state.values():
                category = np.array([obj_dict['category']]).astype(np.int64)
                state_np.append(np.concatenate([obj_dict['position'], category], axis=0))
            state_np = np.concatenate(state_np, axis=0)
            states_np.append(state_np)
        return np.stack(states_np, axis=0)
    
    def flatten_actions(self, actions_list):
        """
        input: [action1, action2, ...]
        output: nparr, shape [num_actions, 2*num_objs]
        """
        actions_np = []
        for action in actions_list:
            action_np = []
            for obj_dict in action.values():
                action_np.append(obj_dict['linear_vel'])
            action_np = np.concatenate(action_np, axis=0)
            actions_np.append(action_np)
        return np.stack(actions_np, axis=0)
    
    def unflatten_states(self, states_np):
        """
        input: [state1_np, state2_np, ...]
        output: nparr, shape [state1_dicts_list, ... ]
        """
        states_list = []
        for state_np in states_np:
            state_dicts = OrderedDict()
            for idx in range(self.num_objs):
                obj_dict = {"position": state_np[idx*3:idx*3+2], "category": state_np[idx*3+2].astype(np.int64).item()}
                state_dicts[f'obj{idx}'] = obj_dict
            states_list.append(state_dicts)
        return states_list
    
    def unflatten_actions(self, actions_np):
        """
        input: [action1_np, action2_np, ...]
        output: nparr, shape [action1_dicts_list, ... ]
        """
        actions_list = []
        for action_np in actions_np:
            action_dicts = OrderedDict()
            for idx in range(self.num_objs):
                obj_dict = {"linear_vel": action_np[idx*2:(idx+1)*2]}
                action_dicts[f'obj{idx}'] = obj_dict
            actions_list.append(action_dicts)
        return actions_list

    def nop_action(self):
        """
        action space (OrderedDict): {'obj1': Dict_1("linear_vel": Box(2)), ..., 'obj_{num_objs}': Dict_{num_objs}("linear_vel": Box(2))}
        """
        action = OrderedDict()
        for idx in range(self.num_objs):
            action[f'obj{idx}'] = {"linear_vel": np.zeros((2, ))}
        return action

    def nop(self):
        """
        no operation at this time step
        """
        action = self.nop_action()
        action_flatten = self.flatten_action([action])
        return self.step(action_flatten[0])

    def pseudo_likelihoods(self, states_list):
        """
        input: [state1, ..., statek]
        output: a nparr of pseudo likelihoods shape [num_states,] 
        """
        assert isinstance(states_list, list)
        likelihoods = []
        for state in states_list:
            if isinstance(state, OrderedDict):
                assert len(state.values()) == self.num_objs
                state_np = self.flatten_states([state])[0] # state_np.shape == (num_objs, 2), positions only
            else:
                assert isinstance(state, np.ndarray) and state.shape[0] == self.num_objs * 3
                state_np = state
            likelihoods.append(get_pseudo_likelihood(state_np, self.pattern, self.num_per_class, self.catetory_list, self.bound, self.r))
        return np.stack(likelihoods)

    def step(self, action, step_num=4, centralized=False, soft_collision=False):
        """
        action: input action, [action_dict_1, action_dict_2, ...]
        for each dim,  [-max_vel,  max_vel]
        """
        collision_num = 0
        
        # flatten: dicts_list -> controls
        if isinstance(action, OrderedDict):
            controls = self.flatten_actions([action])[0].reshape((-1, 2)) # [num_objs, 2]
        else:
            assert isinstance(action, np.ndarray)
            controls = action.reshape((-1, 2))
        assert controls.shape[0] == self.num_objs 

        # normalise controls
        max_vel_norm = np.max(np.abs(controls))
        scale_factor = self.max_action / (max_vel_norm+1e-7)
        scale_factor = np.min([scale_factor, 1])
        controls = scale_factor * controls

        # check input-scale of the action
        max_vel = controls.max()
        if max_vel > self.max_action:
            print(f'!!!!!current max velocity {max_vel} exceeds max action {self.max_action}!!!!!')
        
        # simulate velocities and calc collision num
        self.apply_control(controls, self.balls_list)
        for _ in range(step_num):
            p.stepSimulation(physicsClientId=self.cid)
            collision_num += self.get_collision_num(self.balls_list, centralized)
        collision_num /= step_num 
        if not soft_collision:
            collision_num = (collision_num > 0) 

        # judge if is done
        self.cur_steps += 1
        is_done = (self.cur_steps >= self.max_episode_len)

        # current state
        cur_state = self.get_state(self.balls_list)

        # calc reward, currently it is defined as delta-likelihood
        r = self.pseudo_likelihoods([cur_state]) - self.pseudo_likelihoods([self.prev_state])
        
        infos = {'collision_num': collision_num,
                'is_done': is_done, 
                'init_state': self.init_state, 
                'progress': self.cur_steps / self.max_episode_len,
                'cur_steps': self.cur_steps, 
                'max_episode_len': self.max_episode_len}
        
        # update prev_state
        self.prev_state = cur_state
        return cur_state, r, is_done, infos

    def reset(self, is_random=True, remove_collision=True):
        self.num_episodes += 1
        # initialise objects
        if is_random:
            balls_dict = sample_example_positions(self.num_per_class, self.catetory_list, pattern='random', bound=self.bound, r=self.r)
        else:  # init to target distribution
            balls_dict = sample_example_positions(self.num_per_class, self.catetory_list, pattern=self.pattern, bound=self.bound, r=self.r)

        for color, positions in balls_dict.items():
            self.balls[color] = self.add_balls(positions, color)
        
        # remove remaining collisions via physical simulation
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
        
        # update episodic information
        self.cur_steps = 0
        self.init_state = self.get_state(self.balls_list)
        self.prev_state = deepcopy(self.init_state)

        return self.init_state

    def sample_action(self):
        """
        sample a random action according to current action space
        each dim ranges in [-self.max_action, self.max_action]
        """
        action_np = np.random.normal(size=(self.num_objs * 2)).clip(-1, 1) * self.max_action
        action = self.unflatten_actions([action_np])[0]
        return action
    
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
