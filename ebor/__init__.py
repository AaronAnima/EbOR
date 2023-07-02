from gym.envs.registration import register
pattern_list = ['Circle', 'Cluster', 'CircleCluster']
category_list = ['Red', 'Green', 'Blue']
max_num_objs = 30

for pattern in pattern_list:
    for num_objs in range(1, 1+max_num_objs):
        for n_color in range(1, len(category_list)+1):
            if num_objs % n_color != 0:
                continue
            if pattern == 'CircleCluster':
                horizon = 300
            else:
                horizon = 100
            register(
                id='{}-{}Ball{}Class-v0'.format(pattern, num_objs, n_color),
                entry_point='ebor.Envs.rearrangement:BallGym',
                max_episode_steps=horizon,
                kwargs={
                    'num_per_class': num_objs//n_color,
                    'category_list': category_list[:n_color],
                    'pattern': pattern,
                    'exp_data': None,
                    'time_freq': 4*50,
                    'is_gui': False,
                    'max_action': 0.3,
                    'max_episode_len': horizon,
                    'action_type': 'vel',
                },
            )

stacking_horizon = 30
pattern = 'Stacking'
register(
    id='{}-{}Box{}Class-v0'.format(pattern, 3, 3),
    entry_point='ebor.Envs.rearrangement:BoxGym',
    max_episode_steps=stacking_horizon,
    kwargs={
        'num_per_class': 3,
        'category_list': category_list[:3],
        'pattern': pattern,
        'exp_data': None,
        'time_freq': 60,
        'is_gui': False,
        'max_episode_len': stacking_horizon,
        'action_type': 'vel',
    },
)

stacking_horizon = 30
pattern = 'Stacking'
register(
    id='{}-{}Box{}Class-v0'.format(pattern, 2, 2),
    entry_point='ebor.Envs.rearrangement:BoxGym',
    max_episode_steps=stacking_horizon,
    kwargs={
        'num_per_class': 2,
        'category_list': category_list[:2],
        'pattern': pattern,
        'exp_data': None,
        'time_freq': 60,
        'is_gui': False,
        'max_episode_len': stacking_horizon,
        'action_type': 'vel',
    },
)

