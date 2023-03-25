from gym.envs.registration import register
pattern_list = ['Circle', 'Cluster', 'CircleCluster']
category_list = ['Red', 'Green', 'Blue']

for pattern in pattern_list:
    for n_boxes in range(1, 31):
        for n_color in range(1, len(category_list)+1):
            if n_boxes % n_color != 0:
                continue
            if pattern == 'CircleCluster':
                horizon = 300
            else:
                horizon = 100
            register(
                id='{}-{}Ball{}Class-v0'.format(pattern, n_boxes, n_color),
                entry_point='ebor.Envs.Rearrangement:BallGym',
                max_episode_steps=horizon,
                kwargs={
                    'n_boxes': n_boxes//n_color,
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

# # ===== Circling =====
# register(
#     id='Circle-v0',
#     entry_point='ebor.Envs.Rearrangement:BallGym',
#     max_episode_steps=100,
#     kwargs={
#         'n_boxes': 21,
#         'category_list': ['red', ],
#         'pattern': 'Circle',
#         'exp_data': None,
#         'time_freq': 4*50,
#         'is_gui': False,
#         'max_action': 0.3,
#         'max_episode_len': 100,
#         'action_type': 'vel',
#     },
# )
#
# # ===== Clustering =====
# # horizon 100
# register(
#     id='Clustering-v0',
#     entry_point='ebor.Envs.Rearrangement:BallGym',
#     max_episode_steps=100,
#     kwargs={
#         'n_boxes': 7,
#         'category_list': ['red', 'green', 'blue'],
#         'pattern': 'Cluster',
#         'exp_data': None,
#         'time_freq': 4*50,
#         'is_gui': False,
#         'max_action': 0.3,
#         'max_episode_len': 100,
#         'action_type': 'vel',
#     },
# )
#
# # ===== Hybrid of Circling+Clustering =====
# # horizon 300
# register(
#     id='CircleCluster-v0',
#     entry_point='ebor.Envs.Rearrangement:BallGym',
#     max_episode_steps=300,
#     kwargs={
#         'n_boxes': 7,
#         'category_list': ['red', 'green', 'blue'],
#         'pattern': 'CircleCluster',
#         'exp_data': None,
#         'time_freq': 4*50,
#         'is_gui': False,
#         'max_action': 0.3,
#         'max_episode_len': 300,
#         'action_type': 'vel',
#     },
# )
#

