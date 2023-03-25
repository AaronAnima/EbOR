from gym.envs.registration import register

# ===== Circling =====
register(
    id='Circling-v0',
    entry_point='ebor.Envs.Rearrangement:BallGym',
    max_episode_steps=100,
    kwargs={
        'n_boxes': 21,
        'category_list': ['red'],
        'pattern': 'circle',
        'exp_data': None,
        'time_freq': 4*50,
        'is_gui': False,
        'max_action': 0.3,
        'max_episode_len': 100,
        'action_type': 'vel',
    },
)

# ===== Clustering =====
# horizon 100
register(
    id='Clustering-v0',
    entry_point='ebor.Envs.Rearrangement:BallGym',
    max_episode_steps=100,
    kwargs={
        'n_boxes': 7,
        'category_list': ['red', 'green', 'blue'],
        'pattern': 'clustering',
        'exp_data': None,
        'time_freq': 4*50,
        'is_gui': False,
        'max_action': 0.3,
        'max_episode_len': 100,
        'action_type': 'vel',
    },
)

# ===== Hybrid of Circling+Clustering =====
# horizon 300
register(
    id='CirclingClustering-v0',
    entry_point='ebor.Envs.Rearrangement:BallGym',
    max_episode_steps=300,
    kwargs={
        'n_boxes': 7,
        'category_list': ['red', 'green', 'blue'],
        'pattern': 'circle_cluster',
        'exp_data': None,
        'time_freq': 4*50,
        'is_gui': False,
        'max_action': 0.3,
        'max_episode_len': 300,
        'action_type': 'vel',
    },
)


