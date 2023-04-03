from gym.envs.registration import register
# pattern combination:
shape_list = ['Circle', 'Triangle', '']
color_list = ['AABB', 'ABAB', 'Random']
# max categories / num objects
category_list = ['Red', 'Green', 'Blue']
max_num_objs = 30
# max_timesteps
horizon = 300

for shape_pattern in shape_list:
    for color_pattern in color_list:
        pattern = f'{shape_pattern}-{color_pattern}'
        for num_objs in range(1, 1+max_num_objs):
            for n_color in range(1, len(category_list)+1):
                if num_objs % n_color != 0:
                    continue
                register(
                    id='{}-{}Ball{}Class-v0'.format(pattern, num_objs, n_color),
                    entry_point='ebor.Envs.Rearrangement:BallGym',
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
