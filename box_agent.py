import gym
import ebor
import cv2
import numpy as np
import argparse
from ipdb import set_trace
from collections import OrderedDict
from tqdm import tqdm


class BoxStackingPlanner:
    def __init__(self, r, bound):
        self.r = r
        self.bound = bound
    
    def get_action_plan(self, init, goal, fps=10):
        actions = []
        # for obj_dict in goal.values():
        #     print(obj_dict['position'])
        #     print(obj_dict['category'])
        # for obj_dict in init.values():
        #     print(obj_dict['position'])
        #     print(obj_dict['category'])
        # set_state()
        for key_init, key_goal in zip(init.keys(), goal.keys()):
            assert key_init == key_goal
            init_dict = init[key_init]
            goal_dict = goal[key_goal]
            # comp action
            pick = init_dict['position'][:2]
            place = goal_dict['position']
            # interpolate
            places = np.linspace(place + np.array([0, 0, 0.1]), place, fps)
            id_number = "".join(list(filter(str.isdigit, key_init)))
            obj_id = np.array([int(id_number)], dtype=np.int64)
            # print(obj_id)
            # print(pick)
            # print(place)
            for item in places:
                action = np.concatenate([pick, item, obj_id], axis=0)
                # print(action[2:2+3])
                actions.append(action)
                pick = item[:2]
        return actions

def images_to_video(path, images, fps, size):
    out = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(size, size), isColor=True)
    for item in images:
        out.write(item)
    out.release()

IS_GUI = False
# IS_GUI = True
FPS = 5
IM_SIZE = 1024
num_videos = 100
# pattern = 'ClusterStacking'
pattern = 'InterlaceStacking'
num_per_class = 4
class_num = 3
env = gym.make(f'{pattern}-{num_per_class*class_num}Box{class_num}Class-v0', is_gui=IS_GUI)
planner = BoxStackingPlanner(env.r, env.bound)

for video_idx in tqdm(range(num_videos)):
    goal_state = env.reset(is_random=False)
    cv2.imwrite('goal.png', env.render(img_size=IM_SIZE))
    init_state = env.reset(is_random=True)
    cv2.imwrite('init.png', env.render(img_size=IM_SIZE))
    actions = planner.get_action_plan(init_state, goal_state, fps=FPS)

    video = []
    for idx, action in enumerate(actions):
        env.step(action)
        cur_img = env.render(img_size=IM_SIZE)
        video.append(cur_img)
    images_to_video(path=f'./videos/{video_idx}_vidoe.mp4', images=video, fps=FPS, size=IM_SIZE)




# env = gym.make(args.env_id)   # choose the environment
# # env = LoadDataset(env, 'data/clustering/clustering_7_1000.pkl') # an example of loading the expert dataset
# state = env.reset(is_random=False)  # if is_random=False, the env will reset to a target example state
# if args.render:
#     cv2.imshow('target', env.render())   # show the target image
#     cv2.waitKey(1)

# while True:
#     if args.render:
#         state = env.reset(is_random=False)  # if is_random=False, the env will reset to a target example state
#         cv2.imshow('target', env.render())  # show the target image
#         cv2.waitKey(1)
#     done = False
#     state = env.reset()
#     while not done: # get done in 100 steps
#         random_action = env.action_space.sample() # get a random action
#         state, reward, done, info = env.step(random_action) # take a step
#         if args.render:
#             cv2.imshow('img', env.render()) # show the image
#             cv2.waitKey(1)


