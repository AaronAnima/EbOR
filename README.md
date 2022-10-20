# EbOR
EbOR: environments for Example-based Object Rearrangement

Currently there are three environments in this repo:

| *Circling-v0* | *Clustering-v0* | *CirclingClustering-v0* |
|  ----  | ----  | ----  | 
|<img src="demos/circling_demo.gif" align="middle" width="230"/>  | <img src="demos/clustering_demo.gif" align="middle" width="230"/>  | <img src="demos/hybrid_demo.gif" align="middle" width="230"/>    |

# Install

## Requirements
- Ubuntu >= 18.04
- python >= 3.6
- gym>=0.20.0,<0.25.0a0
- pybullet >= 3.2.5

## Installation
```
git clone https://github.com/AaronAnima/EbOR

cd EbOR

pip install -e .
```

# Getting Started
```
import gym
import ebor
# import cv2

env = gym.make('CirclingClustering-v0')
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
```

## Citation
```
@inproceedings{wu2022targf,
  title     = {Tar{GF}: Learning Target Gradient Field for Object Rearrangement},
  author    = {Mingdong Wu and fangwei zhong and Yulong Xia and Hao Dong},
  booktitle = {Thirty-Sixth Conference on Neural Information Processing Systems},
  year      = {2022},
  url       = {https://openreview.net/forum?id=Euv1nXN98P3}
}

```

## Contact
If you have any suggestion or questions, please get in touch at [wmingd@pku.edu.cn](wmingd@pku.edu.cn) or [zfw1226@gmail.com](zfw1226@gmail.com).

## LICENSE
TarGF has an MIT license, as found in the [LICENSE](./LICENSE) file.
