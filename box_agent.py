import gym
import ebor
import cv2
import numpy as np
import argparse
from ipdb import set_trace
from collections import OrderedDict


class BoxStackingPlanner:
    def __init__(self, r, bound):
        self.r = r
        self.bound = bound
    
    def get_plan(self, init, goal):
        actions = []
        for key_init, key_goal in zip(init.keys(), goal.keys()):
            assert key_init == key_goal
            init_dict = init[key_init]
            goal_dict = goal[key_goal]
            # comp action
            pick = init_dict['position'][:2]
            place = goal_dict['position']
            obj_id = np.array([int(key_init[-1:])], dtype=np.int64)
            action = np.concatenate([pick, place, obj_id], axis=0)
            actions.append(action)
        return actions

def images_to_video(path, images, fps, size):
    out = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(size, size), isColor=True)
    for item in images:
        out.write(item)
    out.release()

# env = gym.make('Stacking-3Box3Class-v0')
env = gym.make('Stacking-2Box2Class-v0')
planner = BoxStackingPlanner(env.r, env.bound)
goal_state = env.reset(is_random=False)
cv2.imwrite('goal.png', env.render(img_size=256))
init_state = env.reset(is_random=True)
cv2.imwrite('init.png', env.render(img_size=256))
actions = planner.get_plan(init_state, goal_state)
# set_trace()

video = []
for action in actions:
    env.step(action)
    cur_img = env.render(img_size=256)
    video.append(cur_img)
    cv2.imwrite('debug.png', env.render(img_size=256))
    set_trace()
images_to_video(path='debug.mp4', images=video, fps=1, size=256)




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


