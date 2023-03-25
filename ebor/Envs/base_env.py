import time
import pickle
import random
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import pybullet as p
import os
import sys

class BallEnv(gym.Env):
    def __init__(self, max_episode_len=250, category_list=['red', 'green', 'blue'], is_gui=False, time_freq=240, wall_bound=0.3, action_type='vel', **kwargs):
        n_boxes = kwargs['n_boxes']
        self.catetory_list = category_list
        self.num_class = len(self.catetory_list)
        self.action_type = action_type

        if is_gui:
            self.cid = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        else:
            self.cid = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        my_data_path = os.path.join(os.path.dirname(__file__), 'Assets')
        p.setAdditionalSearchPath(my_data_path)  # optionally

        # first set a base plane
        self.plane_base = p.loadURDF("plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
                                     physicsClientId=self.cid)
        self.n_boxes_per_class = n_boxes

        # set gravity
        p.setGravity(0, 0, -10, physicsClientId=self.cid)

        # set time step
        p.setTimeStep(1. / time_freq, physicsClientId=self.cid)
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
        self.balls = {key: [] for key in self.catetory_list}

        self.name_mapping_urdf = {'red': f"sphere_red_{action_type}.urdf", 'green': f"sphere_green_{action_type}.urdf",
                                  'blue': f"sphere_blue_{action_type}.urdf"}

        self.n_boxes = self.n_boxes_per_class * self.num_class # assume each class of the same number of balls
        # self.balls = []

        self.max_episode_len = max_episode_len
        self.num_episodes = 0

        # reset cam-pose
        if is_gui:
            # reset cam-pose to a top-down view
            p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0., cameraPitch=-89.,
                                         cameraTargetPosition=[0, 0, 0], physicsClientId=self.cid)
        self.balls_list = []
        self.exp_data = None
        self.init_state = None

    def add_balls(self, positions, category='red'):
        """
        load balls at given positions
        category in ['red', 'green', 'blue']
        positions: [n_balls_per_class, 2] # 2-d coordinates
        """
        cur_list = self.balls[category] # get the list of loaded balls with the given category
        flag_load = (len(cur_list) == 0) # if the list is empty, then load the balls
        cur_urdf = self.name_mapping_urdf[category] # get the urdf file name
        iter_list = range(self.n_boxes_per_class) if flag_load else cur_list
        radius_list = [1.0, 1.5, 2.0]
        for i, item in enumerate(iter_list):
            horizon_p = positions[i]
            horizon_p = np.clip(horizon_p, -(self.bound - self.r), (self.bound - self.r))
            cur_ori = p.getQuaternionFromEuler([0, 0, 0])
            if flag_load:
                cur_list.append(p.loadURDF(cur_urdf, [horizon_p[0].item(), horizon_p[1].item(), self.r], cur_ori, physicsClientId=self.cid))
            else:
                p.resetBasePositionAndOrientation(item, [horizon_p[0].item(), horizon_p[1].item(), self.r], cur_ori, physicsClientId=self.cid)
                p.resetBaseVelocity(item, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        return  cur_list
    def set_state(self, state, obj_list, verbose=None):
        """
        set 2-d positions for each object
        state: [3*n_balls_per_class, 2]
        """
        # verbose is to fit the api
        assert state.shape[0] == len(obj_list) * 2
        for idx, boxId in enumerate(obj_list):
            cur_state = state[idx * 2:(idx + 1) * 2]
            # un-normalize
            cur_pos = (self.bound - self.r) * cur_state
            cur_ori = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(boxId, [cur_pos[0], cur_pos[1], 0], cur_ori, physicsClientId=self.cid)
            p.resetBaseVelocity(boxId, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        return self.get_state()

    def check_valid(self):
        """
        check whether all objects are physically correct(no floating balls)
        """
        positions = []
        for ballId in self.balls_list:
            pos, ori = p.getBasePositionAndOrientation(ballId, physicsClientId=self.cid)
            positions.append(pos)
        positions = np.stack(positions)

        flag_x_bound = np.max(positions[:, 0:1]) <= (self.bound - self.r) and np.min(positions[:, 0:1]) >= (
                    -self.bound + self.r)
        flag_y_bound = np.max(positions[:, 1:2]) <= (self.bound - self.r) and np.min(positions[:, 1:2]) >= (
                    -self.bound + self.r)

        flag_height = np.max(np.abs(positions[:, -1:] - self.r)) < 0.001
        return flag_height & flag_x_bound & flag_y_bound, (positions[:, -1:])

    def get_state(self, obj_list, norm=True):
        """
        get 2-d positions for each object
        Input: obj_list: [n_balls]
        return: [3*n_balls_per_class*2]
        if norm, then normalize each 2-d position to [-1, 1]
        """
        box_states = []
        for boxId in (obj_list):
            pos, ori = p.getBasePositionAndOrientation(boxId, physicsClientId=self.cid)

            pos = np.array(pos[0:2], dtype=np.float32)

            # normalize -> [-1, 1]
            if norm:
                pos = pos / (self.bound - self.r)

            box_state = pos
            box_states.append(box_state)
        box_states = np.concatenate(box_states, axis=0)
        assert box_states.shape[0] == self.n_boxes * 2
        return box_states

    def set_velocity(self, vels, obj_list):
        """
        set 2-d linear velocity for each object
        vels: [3*n_balls_per_class, 2]
        obj_list: [boxID]
        """
        # vels.shape = [num_boxes, 2]
        # set_trace()
        for boxId, vel in zip(obj_list, vels):
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

    def get_collision_num(self, obj_list, centralized=True):
        """
        return the collision number at current step
        collision_num = sum_{1 <= i < j <= K} is_collision(object_i, object_j)
        """
        # collision detection
        items = obj_list
        collisions = np.zeros((len(items), len(items)))
        for idx1, ball_1 in enumerate(items[:-1]):
            for idx2, ball_2 in enumerate(items[idx1 + 1:]):
                points = p.getContactPoints(ball_1, ball_2, physicsClientId=self.cid)
                collisions[idx1][idx2] = (len(points) > 0)
                # for debug
                # print(f'{name1} {name2} {len(points)}')
        return np.sum(collisions).item() if centralized else collisions

    def calc_reward(self):
        """
        return a scalar to measure the plausibility of sorting
        r = intra_var*intra_scale # intra variance of each class
          + corss_scale*center_var # inter-variance of each pair of classes
          + c_dist_scale*(r_c_dist+g_c_dist+b_c_dist) # Is the distance between each center and the origin correct
        """
        cur_state = self.get_state()

        # split states and calc each center
        red_states = np.reshape(cur_state[0:2 * self.n_boxes_per_class], (self.n_boxes_per_class, 2))
        green_states = np.reshape(cur_state[2 * self.n_boxes_per_class:4 * self.n_boxes_per_class],
                                  (self.n_boxes_per_class, 2))
        blue_states = np.reshape(cur_state[4 * self.n_boxes_per_class:6 * self.n_boxes_per_class],
                                 (self.n_boxes_per_class, 2))
        red_center = np.mean(red_states, axis=0)[None, :]
        green_center = np.mean(green_states, axis=0)[None, :]
        blue_center = np.mean(blue_states, axis=0)[None, :]

        # calc intra-var
        radius = 0.18
        r_c_dist = np.abs(np.sqrt(np.sum(red_center ** 2)) - radius)
        g_c_dist = np.abs(np.sqrt(np.sum(green_center ** 2)) - radius)
        b_c_dist = np.abs(np.sqrt(np.sum(blue_center ** 2)) - radius)
        red_var = np.mean(np.sum((red_states - red_center) ** 2, axis=-1), axis=0)
        green_var = np.mean(np.sum((green_states - green_center) ** 2, axis=-1), axis=0)
        blue_var = np.mean(np.sum((blue_states - blue_center) ** 2, axis=-1), axis=0)
        intra_var = (red_var + green_var + blue_var) / 3

        # calc extra-var
        centers = np.stack((red_center, green_center, blue_center), axis=0)
        mean_center = np.mean(centers, axis=0)[None, :]
        center_var = np.mean(np.sum((centers - mean_center) ** 2, axis=-1), axis=0)[0]

        # sum and balance   reward
        intra_scale = -1.0
        c_dist_scale = -1.0
        corss_scale = 5.0
        return intra_var.item() * intra_scale + corss_scale * center_var.item() + c_dist_scale * (
                    r_c_dist + g_c_dist + b_c_dist).item()

    def change_dynamics(self, obj_list):
        for body_id in obj_list:
            p.changeDynamics(body_id, -1, lateralFriction=1., spinningFriction=1, rollingFriction=1,
                             restitution=0.1, physicsClientId=self.cid)
        p.changeDynamics(self.plane_base, -1, lateralFriction=0., spinningFriction=0, rollingFriction=0,
                         restitution=0.1, physicsClientId=self.cid)

    def close(self):
        p.disconnect(physicsClientId=self.cid)
