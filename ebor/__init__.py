from gym.envs.registration import register

# ===== Circling =====
register(
    id='Circling-v0',
    entry_point='ebor.Envs.Circling:CirclingGym',
    max_episode_steps=100,
    kwargs={
        'n_boxes': 7,
        'exp_data': None,
        'time_freq': 200,
        'is_gui': False,
        'max_action': 0.3,
        'max_episode_len': 100,
        'action_type': 'vel',
    },
)

# ===== Clustering =====
register(
    id='Clustering-v0',
    entry_point='ebor.Envs.Clustering:ClusteringGym',
    max_episode_steps=100,
    kwargs={
        'n_boxes': 7,
        'exp_data': None,
        'time_freq': 200,
        'is_gui': False,
        'max_action': 0.3,
        'max_episode_len': 100,
        'action_type': 'vel',
    },
)

# ===== Hybrid of Circling+Clustering =====
register(
    id='CirclingClustering-v0',
    entry_point='ebor.Envs.CirclingClustering:CirclingClusteringGym',
    max_episode_steps=100,
    kwargs={
        'n_boxes': 7,
        'exp_data': None,
        'time_freq': 200,
        'is_gui': False,
        'max_action': 0.3,
        'max_episode_len': 300,
        'action_type': 'vel',
    },
)

