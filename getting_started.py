import gym
import ebor
# import cv2

env = gym.make('Clustering-v0')
x = env.reset() # to init objects

while True:
    done = False
    x = env.reset(is_random=False) # if is_random=False, the env will reset to a target example state

    img = env.render()
    # cv2.imwrite('./target_example.png', img)

    x = env.reset()
    while not done:
        random_action = env.action_space.sample()
        x, reward, done, truncated, info = env.step(random_action)

        img = env.render()
        # cv2.imwrite('./in_process.png', img)
