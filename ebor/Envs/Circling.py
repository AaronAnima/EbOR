import time
import pickle
import random
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import pybullet as p
import sys
import os

class CirclingSimulator(gym.Env):
    def __init__(self, max_episode_len=250, is_gui=False, time_freq=240, wall_bound=0.3, action_type='vel', **kwargs):
        n_boxes = kwargs['n_boxes']
        self.action_type = action_type

        if is_gui:
            self.cid = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        else:
            self.cid = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        my_data_path = os.path.join(os.path.dirname(__file__), 'Assets')
        p.setAdditionalSearchPath(my_data_path)  # optionally

        # first set a base plane
        self.plane_base = p.loadURDF("plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.cid)
        self.n_boxes_per_class = n_boxes

        # set gravity
        p.setGravity(0, 0, -10, physicsClientId=self.cid)

        # set time step
        p.setTimeStep(1./time_freq, physicsClientId=self.cid)
        self.time_freq = time_freq

        # then set 4 transparent planes surrounded
        ori1 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori2 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori3 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        ori4 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        pos1 = [wall_bound, 0, 0]
        pos2 = [-wall_bound, 0, 0]
        pos3 = [0, wall_bound, 0]
        pos4 = [0, -wall_bound, 0]
        self.bound = wall_bound
        self.r = 0.025
        plane_name = "plane_transparent.urdf"
        scale = wall_bound / 2.5
        self.transPlane1 = p.loadURDF(plane_name, pos1, ori1, globalScaling=scale, physicsClientId=self.cid)
        self.transPlane2 = p.loadURDF(plane_name, pos2, ori2, globalScaling=scale, physicsClientId=self.cid)
        self.transPlane3 = p.loadURDF(plane_name, pos3, ori3, globalScaling=scale, physicsClientId=self.cid)
        self.transPlane4 = p.loadURDF(plane_name, pos4, ori4, globalScaling=scale, physicsClientId=self.cid)

        # init ball list for R,G,B balls
        self.name_mapping_urdf = {'red': f"sphere_red_{action_type}.urdf", 'green': f"sphere_green_{action_type}.urdf", 'blue': f"sphere_blue_{action_type}.urdf"}

        self.n_boxes = self.n_boxes_per_class * 3
        self.balls = []

        self.max_episode_len = max_episode_len
        self.num_episodes = 0

        # reset cam-pose
        if is_gui:
            # reset cam-pose to a top-down view
            p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0., cameraPitch=-89., cameraTargetPosition=[0, 0, 0], physicsClientId=self.cid)

    def add_balls(self, positions):
        cur_list = self.balls
        flag_load = (len(cur_list) == 0)
        cur_urdf = self.name_mapping_urdf['red']
        iter_list = range(self.n_boxes) if flag_load else cur_list
        for i, item in enumerate(iter_list):
            horizon_p = positions[i]
            horizon_p = np.clip(horizon_p, -(self.bound-self.r), (self.bound-self.r))
            cur_ori = p.getQuaternionFromEuler([0, 0, 0])
            if flag_load:
                self.balls.append(p.loadURDF(cur_urdf, [horizon_p[0].item(), horizon_p[1].item(), self.r], cur_ori, physicsClientId=self.cid))
            else:
                # set_trace()
                p.resetBasePositionAndOrientation(item, [horizon_p[0].item(), horizon_p[1].item(), self.r], cur_ori, physicsClientId=self.cid)
                p.resetBaseVelocity(item, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
    
    def get_positions(self, is_random=True, is_permutation=True, dynamic_center=True):
        """
        sample i.i.d. gaussian 2-d positions centered on 'center'
        return positions: [n_boxes_per_class, 2]
        """
        if not is_random:
            ''' sample radius, center '''
            r_min = self.r / np.sin(np.pi/len(self.balls))
            r_max = self.bound - self.r
            assert r_max > r_min
            center_bound = self.bound - self.r - r_min
            if dynamic_center:
                center = np.random.uniform(-center_bound, center_bound, size=(2, )).reshape(-1, 2)
                center = np.repeat(center, self.n_boxes, axis=0)
            else:
                center = np.zeros((self.n_boxes, 2))
            r_max_ = self.bound - self.r - np.max(np.abs(center))
            if not dynamic_center:
                mu = (r_max_+r_min)/2
                std = 0.01
                r_sampled = np.random.normal(size=(self.n_boxes, )) * std + mu
            else:
                r_sampled = np.random.uniform(r_min, r_max_, size=(1, ))
                r_sampled = np.repeat(r_sampled, self.n_boxes, axis=0)
            r_sampled = np.clip(r_sampled, r_min, r_max_)


            ''' sample theta '''
            thetas = np.array(range(self.n_boxes)) * (2*np.pi/self.n_boxes)

            if is_permutation:
                permutation = np.random.permutation(self.n_boxes)
                thetas = thetas[permutation]
                r_sampled = r_sampled[permutation]

            positions = np.concatenate([(r_sampled*np.cos(thetas)).reshape((-1, 1)),
                                        (r_sampled*np.sin(thetas)).reshape((-1, 1))], axis=-1)
            positions += center
        else:
            scale = self.bound - self.r
            positions = np.random.uniform(-1, 1, size=(self.n_boxes, 2)) * scale
        return positions

    def set_state(self, state, verbose=None):
        """
        set 2-d positions for each object
        state: [3*n_balls_per_class, 2]
        """
        # verbose is to fit the api
        assert state.shape[0] == len(self.balls) * 2
        for idx, boxId in enumerate(self.balls):
            cur_state = state[idx * 2:(idx + 1) * 2]
            # un-normalize
            cur_pos = (self.bound-self.r) * cur_state
            cur_ori = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(boxId, [cur_pos[0], cur_pos[1], 0], cur_ori, physicsClientId=self.cid)
            p.resetBaseVelocity(boxId, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        return self.get_state()

    def check_valid(self):
        """
        check whether all objects are physically correct(no floating balls)
        """
        positions = []
        for ballId in (self.balls):
            pos, ori = p.getBasePositionAndOrientation(ballId, physicsClientId=self.cid)
            positions.append(pos)
        positions = np.stack(positions)

        flag_x_bound = np.max(positions[:, 0:1]) <= (self.bound-self.r) and np.min(positions[:, 0:1]) >= (-self.bound+self.r)
        flag_y_bound = np.max(positions[:, 1:2]) <= (self.bound-self.r) and np.min(positions[:, 1:2]) >= (-self.bound+self.r)

        flag_height = np.max(np.abs(positions[:, -1:] - self.r)) < 0.001
        return flag_height&flag_x_bound&flag_y_bound, (positions[:, -1:])

    def get_state(self, norm=True):
        """
        get 2-d positions for each object
        return: [3*n_balls_per_class*2]
        if norm, then normalize each 2-d position to [-1, 1]
        """
        box_states = []
        for boxId in self.balls:
            pos, ori = p.getBasePositionAndOrientation(boxId, physicsClientId=self.cid)

            pos = np.array(pos[0:2], dtype=np.float32)

            # normalize -> [-1, 1]
            if norm:
                pos = pos / (self.bound - self.r)

            box_state = pos
            box_states.append(box_state)
        box_states = np.concatenate(box_states, axis=0)
        assert box_states.shape[0] == len(self.balls) * 2
        return box_states

    def set_velocity(self, vels):
        """
        set 2-d linear velocity for each object
        vels: [3*n_balls_per_class, 2]
        """
        # vels.shape = [num_boxes, 2]
        # set_trace()
        for boxId, vel in zip(self.balls, vels):
            vel = [vel[0].item(), vel[1].item(), 0]
            # print(vel)
            # set_trace()
            if self.action_type == 'vel':
                p.resetBaseVelocity(boxId, linearVelocity=vel, physicsClientId=self.cid)
            else:
                assert self.action_type == 'force'
                pos, _ = p.getBasePositionAndOrientation(boxId, physicsClientId=self.cid)
                p.applyExternalForce(
                    objectUniqueId=boxId,
                    linkIndex=-1,
                    forceObj=vel,
                    posObj=pos,
                    flags=p.WORLD_FRAME,
                    physicsClientId=self.cid
                )

    def render(self, img_size=256, score=None):
        """
        return an  image of  cur state: [img_size, img_size, 3], BGR
        """
        # if grad exists, then add debug line
        viewmatrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0.0, 1.0],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0],
        )
        projectionmatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1,
        )
        _, _, rgba, _, _ = p.getCameraImage(img_size, img_size, viewMatrix=viewmatrix,
                                            projectionMatrix=projectionmatrix, physicsClientId=self.cid)
        rgb = rgba[:, :, 0:3]

        return rgb
    
    def get_collision_num(self, centralized=True):
        """
        return the collision number at current step
        collision_num = sum_{1 <= i < j <= K} is_collision(object_i, object_j)
        """
        # collision detection
        items = self.balls
        collisions = np.zeros((len(items), len(items)))
        for idx1, ball_1 in enumerate(items[:-1]):
            for idx2, ball_2 in enumerate(items[idx1+1:]):
                points = p.getContactPoints(ball_1, ball_2, physicsClientId=self.cid)
                collisions[idx1][idx2] = (len(points) > 0)
                # for debug
                # print(f'{name1} {name2} {len(points)}')
        return np.sum(collisions).item() if centralized else collisions

    @staticmethod
    def sample_action():
        raise NotImplementedError

    def reset(self, is_random=False):
        raise NotImplementedError

    def step(self, vels, duration=10):
        raise NotImplementedError

    def close(self):
        p.disconnect(physicsClientId=self.cid)


class CirclingGym(CirclingSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_action = kwargs['max_action']
        self.action_space = spaces.Box(-self.max_action, self.max_action, shape=(2*3*self.n_boxes_per_class, ), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(2*3*self.n_boxes_per_class, ), dtype=np.float32)
        self.cur_steps = 0

        # load exp_data, for RCE
        exp_data = kwargs['exp_data']
        if exp_data:
            data_path = f'../ExpertDatasets/{exp_data}.pth'
            with open(data_path, 'rb') as f:
                data_samples = pickle.load(f)
            self.exp_data = data_samples
        else:
            self.exp_data = None

        self.init_state = self.reset(is_random=True)
        self._seed()

    def nop(self):
        """
        no operation at this time step
        """
        action = np.zeros((self.n_boxes, 2))
        return self.step(action)

    def step(self, vels, step_size=4, centralized=False, soft_collision=False):
        """
        vels: [3*n_balls_per_class, 2], 2-d linear velocity
        for each dim,  [-max_vel,  max_vel]
        """
        # action: numpy, [num_box*2]
        collision_num = 0
        old_state = self.get_state()
        old_pos = old_state.reshape(self.n_boxes, 2)

        vels = np.reshape(vels, (self.n_boxes, 2)) 
        max_vel_norm = np.max(np.abs(vels))
        scale_factor = self.max_action / (max_vel_norm+1e-7)
        scale_factor = np.min([scale_factor, 1])
        vels = scale_factor * vels
        max_vel = vels.max()
        if max_vel > self.max_action:
            print(f'!!!!!current max velocity {max_vel} exceeds max action {self.max_action}!!!!!')
        self.set_velocity(vels)
        for _ in range(step_size):
            p.stepSimulation(physicsClientId=self.cid)
            collision_num += self.get_collision_num(centralized=centralized)
        collision_num /= step_size 

        ''' M3D20 modify '''
        if not soft_collision:
            collision_num = (collision_num > 0)


        r = 0
        # judge if is done
        self.cur_steps += 1
        is_done = self.cur_steps >= self.max_episode_len

        new_pos = self.get_state().reshape(3*self.n_boxes_per_class, 2)
        delta_pos = new_pos - old_pos
        vel_err = np.max(np.abs(delta_pos*self.time_freq*self.bound/step_size - vels))/self.max_action
        vel_err_mean = np.mean(np.abs(delta_pos*self.time_freq*self.bound/step_size - vels))/self.max_action
        infos = {'delta_pos': delta_pos, 'collision_num': collision_num,
                'vel_err': vel_err, 'vel_err_mean': vel_err_mean,
                'is_done': is_done, 'progress': self.cur_steps / self.max_episode_len,
                'init_state': self.init_state, 'cur_steps': self.cur_steps, 'max_episode_len': self.max_episode_len}

        return self.get_state(), r, is_done, infos

    def reset(self, is_random=True):
        self.num_episodes += 1
        t_s = time.time()
        positions = self.get_positions(is_random)
        
        self.add_balls(positions)

        p.stepSimulation(physicsClientId=self.cid)
        total_steps = 0
        while self.get_collision_num() > 0:
            for _ in range(20):
                p.stepSimulation(physicsClientId=self.cid)
            total_steps += 20
            if total_steps > 10000:
                print('Warning! Reset takes too much trial!')
                break
        # set_trace()
        self.cur_steps = 0
        # print(f'No.{self.num_episodes} Episodes, reset now! time cost: {time.time() - t_s}')
        self.init_state = self.get_state()
        return self.init_state

    def get_dataset(self, num_obs=256):
        """
        to fit the need of RCE baseline
        """
        # sample a batch of random actions
        action_vec = [self.sample_action() for _ in range(num_obs)]
        # sample a batch of example states
        ind = np.random.randint(0, len(self.exp_data), size=(num_obs,))
        obs_vec = self.exp_data[ind]
        dataset = {
            'observations': np.array(obs_vec, dtype=np.float32),
            'actions': np.array(action_vec, dtype=np.float32),
            'rewards': np.zeros(num_obs, dtype=np.float32),
        }
        return dataset

    def sample_action(self):
        """
        sample a random action according to current action space
        return range [-self.max_action, self.max_action]
        """
        return np.random.normal(size=3*self.n_boxes_per_class * 2).clip(-1, 1) * self.max_action

    
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
        return self.get_state()
