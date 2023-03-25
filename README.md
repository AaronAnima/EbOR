# EbOR
EbOR: Environments for Example-based Object Rearrangement

Currently there are three environments in this repo:

| *Circle-21Ball1Class-v0* | *Cluster-21Ball3Class-v0* | *CircleCluster-21Ball3Class-v0* |
|  ----  | ----  | ----  | 
|<img src="demos/circling_demo.gif" align="middle" width="230"/>  | <img src="demos/clustering_demo.gif" align="middle" width="230"/>  | <img src="demos/hybrid_demo.gif" align="middle" width="230"/>    |

# Install

## Requirements
- Ubuntu >= 18.04
- python >= 3.6
- gym>=0.20.0,<0.25.0a0
- pybullet >= 3.2.5
- opencv-python >= 4.6.0

## Installation
```
git clone https://github.com/AaronAnima/EbOR

cd EbOR

pip install -e .
```

# Getting Started
Launch the environment and run a random agent to see the environment in action:
```
python random_agent.py --render
```
Minimal example:
```
import gym
import ebor
import cv2

env = gym.make('CircleCluster-21Ball3Class-v0') # choose the environment
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
```
The environment 
## Citation
```
@inproceedings{
wu2022targf,
title={Tar{GF}: Learning Target Gradient Field for Object Rearrangement},
author={Mingdong Wu and Fangwei Zhong and Yulong Xia and Hao Dong},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=Euv1nXN98P3}
}
```

## Contact
If you have any suggestion or questions, please get in touch at [wmingd@pku.edu.cn](wmingd@pku.edu.cn) or [zfw@pku.edu.cn](zfw@pku.edu.cn)..

## LICENSE
TarGF has an MIT license, as found in the [LICENSE](./LICENSE) file.
