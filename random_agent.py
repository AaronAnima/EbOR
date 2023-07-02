import gym
import ebor
import cv2
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument("-e", "--env_id", nargs='?', default='CircleCluster-21Ball3Class-v0',
                    help='Select the environment to run')
parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
args = parser.parse_args()

env = gym.make(args.env_id)   # choose the environment
# env = LoadDataset(env, 'data/clustering/clustering_7_1000.pkl') # an example of loading the expert dataset
state = env.reset(is_random=False)  # if is_random=False, the env will reset to a target example state
if args.render:
    cv2.imshow('target', env.render())   # show the target image
    cv2.waitKey(1)

while True:
    if args.render:
        state = env.reset(is_random=False)  # if is_random=False, the env will reset to a target example state
        cv2.imshow('target', env.render())  # show the target image
        cv2.waitKey(1)
    done = False
    state = env.reset()
    while not done: # get done in 100 steps
        random_action = env.action_space.sample() # get a random action
        state, reward, done, info = env.step(random_action) # take a step
        if args.render:
            cv2.imshow('img', env.render()) # show the image
            cv2.waitKey(1)


