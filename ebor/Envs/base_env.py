import gym
import numpy as np
import pybullet as p
import os
from ipdb import set_trace
from collections import OrderedDict
from ebor.Envs.constants import BALL_RADIUS, WALL_BOUND

class BallEnv(gym.Env):
    def __init__(self, max_episode_len=250, category_list=['red', 'green', 'blue'], is_gui=False, time_freq=240, wall_bound=0.3, action_type='vel', **kwargs):
        self.catetory_list = category_list
        self.num_classes = len(self.catetory_list)
        self.num_per_class = kwargs['num_per_class']
        self.num_objs = self.num_classes * self.num_per_class
        self.action_type = action_type
        self.max_episode_len = max_episode_len
        self.time_freq = time_freq
        self.bound = WALL_BOUND
        self.r = BALL_RADIUS

        # set physics client
        if is_gui:
            self.cid = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
            # reset cam-pose to a top-down view
            p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0., cameraPitch=-89.,
                                         cameraTargetPosition=[0, 0, 0], physicsClientId=self.cid)
        else:
            self.cid = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version

        # set time step
        p.setTimeStep(1. / time_freq, physicsClientId=self.cid)

        # set gravity
        p.setGravity(0, 0, -10, physicsClientId=self.cid)
        
        # set data path
        my_data_path = os.path.join(os.path.dirname(__file__), 'Assets')
        p.setAdditionalSearchPath(my_data_path)  # optionally

        # first set a base plane
        self.plane_base = p.loadURDF("plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
                                     physicsClientId=self.cid)

        # then set 4 transparent planes surrounded
        ori1 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori2 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori3 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        ori4 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        pos1 = [self.bound, 0, 0]
        pos2 = [-self.bound, 0, 0]
        pos3 = [0, self.bound, 0]
        pos4 = [0, -self.bound, 0]
        plane_name = "plane_transparent.urdf"
        scale = self.bound / 2.5
        
        self.transPlane1 = p.loadURDF(plane_name, pos1, ori1, globalScaling=scale, physicsClientId=self.cid)
        self.transPlane2 = p.loadURDF(plane_name, pos2, ori2, globalScaling=scale, physicsClientId=self.cid)
        self.transPlane3 = p.loadURDF(plane_name, pos3, ori3, globalScaling=scale, physicsClientId=self.cid)
        self.transPlane4 = p.loadURDF(plane_name, pos4, ori4, globalScaling=scale, physicsClientId=self.cid)

        # init ball list for R,G,B balls
        self.balls = {key: [] for key in self.catetory_list}

        self.name_mapping_urdf = {'Red': f"sphere_red_{action_type}.urdf", 'Green': f"sphere_green_{action_type}.urdf",
                                  'Blue': f"sphere_blue_{action_type}.urdf"}

        self.balls_list = []

    def add_balls(self, positions, category='red'):
        """
        load balls at given positions
        category in ['red', 'green', 'blue']
        positions: [num_per_class, 2] # 2-d coordinates
        """
        cur_list = self.balls[category] # get the list of loaded balls with the given category
        flag_load = (len(cur_list) == 0) # if the list is empty, then load the balls
        cur_urdf = self.name_mapping_urdf[category] # get the urdf file name
        iter_list = range(self.num_per_class) if flag_load else cur_list
        for i, item in enumerate(iter_list):
            cur_pos = positions[i]
            cur_pos = np.clip(cur_pos, -(self.bound - self.r), (self.bound - self.r))
            cur_ori = p.getQuaternionFromEuler([0, 0, 0])
            if flag_load:
                cur_list.append(p.loadURDF(cur_urdf, [cur_pos[0].item(), cur_pos[1].item(), self.r], cur_ori, physicsClientId=self.cid))
            else:
                p.resetBasePositionAndOrientation(item, [cur_pos[0].item(), cur_pos[1].item(), self.r], cur_ori, physicsClientId=self.cid)
                p.resetBaseVelocity(item, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        return cur_list

    def get_objs_list(self):
        objs_list = [value for sublist in self.balls.values() for value in sublist]
        return objs_list
    
    def set_state(self, objs_state, obj_list=None, verbose=None):
        """
        objs_state: {"obj1": obj_dict1, "obj2": obj_dict2, ... }
        obj_dicti: {"position": nparr (2, ), "category": nparr (1,)}
        """
        if obj_list is None:
            obj_list = self.get_objs_list()
        assert len(objs_state) == self.num_objs and len(obj_list) == self.num_objs
        for idx, objID in enumerate(obj_list):
            cur_state = objs_state[f'obj{idx}']
            # ensure the category
            gt_category = idx // self.num_per_class
            cur_category = cur_state['category']
            assert gt_category == cur_category
            # un-normalize
            cur_pos = (self.bound - self.r) * cur_state['position']
            cur_ori = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(objID, [cur_pos[0], cur_pos[1], 0], cur_ori, physicsClientId=self.cid)
            p.resetBaseVelocity(objID, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        cur_objs_state = self.get_state()

        # check again
        assert cur_objs_state.keys() == objs_state.keys()
        for key in cur_objs_state:
            cur_obj_dict = cur_objs_state[key]
            inp_obj_dict = objs_state[key]
            assert np.linalg.norm(cur_obj_dict['position'] - inp_obj_dict['position']) < 0.0001
            assert cur_obj_dict['category'] == inp_obj_dict['category']
        return cur_objs_state

    def get_state(self, obj_list=None, norm=True):
        """
        return objs_state: {"obj1": obj_dict1, "obj2": obj_dict2, ... }
        obj_dicti: {"position": nparr (2, ), "category": nparr (1,)}
        if norm, then normalize each 2-d position to [-1, 1]
        """
        if obj_list is None:
            obj_list = self.get_objs_list()
        assert len(obj_list) == self.num_objs
        objs_state = OrderedDict()
        for idx, objID in enumerate(obj_list):
            # get position
            pos, ori = p.getBasePositionAndOrientation(objID, physicsClientId=self.cid)
            pos = np.array(pos[0:2], dtype=np.float32)

            # normalize -> [-1, 1]
            if norm:
                pos = pos / (self.bound - self.r)

            # get category 
            category = idx // self.num_per_class

            obj_state = {"position": pos, "category": category} # pos: nparr (2, ) category: int64
            objs_state[f'obj{idx}'] = obj_state
        assert len(objs_state) == self.num_objs
        return objs_state

    def apply_control(self, controls, obj_list=None):
        """
        set 2-d linear velocity for each object
        controls: [objs_num, 2] # forces or velocities
        obj_list: [objid_1, objid_2, ... ]
        """
        if obj_list is None:
            obj_list = self.get_objs_list()
            
        for objID, vel in zip(obj_list, controls):
            control = [vel[0].item(), vel[1].item(), 0]
            if self.action_type == 'vel':
                p.resetBaseVelocity(objID, linearVelocity=control, physicsClientId=self.cid)
            else:
                assert self.action_type == 'force'
                pos, _ = p.getBasePositionAndOrientation(objID, physicsClientId=self.cid)
                p.applyExternalForce(
                    objectUniqueId=objID,
                    linkIndex=-1,
                    forceObj=control,
                    posObj=pos,
                    flags=p.WORLD_FRAME,
                    physicsClientId=self.cid
                )

    def check_valid(self, obj_list=None):
        """
        check whether all objects are physically correct(no floating balls)
        """
        if obj_list is None:
            obj_list = self.get_objs_list()
        positions = []
        for objID in obj_list:
            pos, ori = p.getBasePositionAndOrientation(objID, physicsClientId=self.cid)
            positions.append(pos)
        positions = np.stack(positions)

        flag_x_bound = np.max(positions[:, 0:1]) <= (self.bound - self.r) and np.min(positions[:, 0:1]) >= (
                    -self.bound + self.r)
        flag_y_bound = np.max(positions[:, 1:2]) <= (self.bound - self.r) and np.min(positions[:, 1:2]) >= (
                    -self.bound + self.r)

        flag_height = np.max(np.abs(positions[:, -1:] - self.r)) < 0.001
        return flag_height & flag_x_bound & flag_y_bound, (positions[:, -1:])

    def render(self, img_size=256, score=None):
        """
        return an  image of  cur state: [img_size, img_size, 3], BGR
        """
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

    def get_collision_num(self, obj_list=None, centralized=True):
        """
        return the collision number at current step
        collision_num = sum_{1 <= i < j <= K} is_collision(object_i, object_j)
        """
        if obj_list is None:
            obj_list = self.get_objs_list()
        collisions = np.zeros((len(obj_list), len(obj_list)))
        for idx1, ball_1 in enumerate(obj_list[:-1]):
            for idx2, ball_2 in enumerate(obj_list[idx1 + 1:]):
                # collision detection
                points = p.getContactPoints(ball_1, ball_2, physicsClientId=self.cid)
                collisions[idx1][idx2] = (len(points) > 0)
        return np.sum(collisions).item() if centralized else collisions

    def change_dynamics(self, obj_list):
        """
        Change Friction-coefficients of the objects
        """
        if obj_list is None:
            obj_list = self.get_objs_list()
        for body_id in obj_list:
            p.changeDynamics(body_id, -1, lateralFriction=1., spinningFriction=1, rollingFriction=1,
                             restitution=0.1, physicsClientId=self.cid)
        p.changeDynamics(self.plane_base, -1, lateralFriction=0., spinningFriction=0, rollingFriction=0,
                         restitution=0.1, physicsClientId=self.cid)

    def close(self):
        p.disconnect(physicsClientId=self.cid)
