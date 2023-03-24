import gym
import ebor
import cv2

env = gym.make('Clustering-v0') # choose the environment
state = env.reset(is_random=False)  # if is_random=False, the env will reset to a target example state
cv2.imshow('target', env.render()) # show the target image
cv2.waitKey(1)
while True:
    done = False
    state = env.reset()
    while not done: # get done in 100 steps
        random_action = env.action_space.sample() # get a random action
        state, reward, done, info = env.step(random_action) # take a step
        img = env.render() # render the image
        cv2.imshow('img', img) # show the image
        cv2.waitKey(1)
