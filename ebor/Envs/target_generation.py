import numpy as np
import random

def sample_example_positions(num_per_class, category_list, pattern, bound, r):
    if pattern == 'Cluster':
        balls_dict = sample_cluster_positions(num_per_class, category_list, bound, r, scale=0.05)
    if pattern == 'Stacking':
        balls_dict = sample_stacking_positions(num_per_class, category_list, bound, r, scale=0.05)
    elif pattern == 'Circle':
        balls_dict = sample_circle_positions(num_per_class, category_list, bound, r, scale=0.05, random_color=True)
    elif pattern == 'CircleCluster':
        balls_dict = sample_circle_positions(num_per_class, category_list, bound, r, scale=0.05, random_color=False)
    else:
        pos_dim = 2
        if '3D' in pattern:
            pos_dim = 3
        balls_dict = sample_random_positions(num_per_class, category_list, bound, r, pos_dim, scale=0.3)
    return balls_dict

# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Clustering ----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

# sample center positions for each class, used in clustering
def sample_centers(num_classes, radius=0.18, random_shuffle=True, random_rotate=True):
    """
    sample centers for each category, note that each pair of centers are equally spaced
    random_shuffle: given c1, determine whether to shuffle the order of c2, c3, ... 
    random_rotate: determine direction angle of r's center(0, 120, 240)
    return red_center:[2], green_center: [2], blue_center: [2]
    """
    # init sample-center for each class
    if random_rotate:
        theta_0 = (np.random.randint(0, num_classes)/num_classes) * 2 * np.pi
    else:
        theta_0 = 0
    deltas = [2 * idx * np.pi / num_classes for idx in range(1, num_classes)]
    if random_shuffle:
        random.shuffle(deltas)
        deltas = [0] + deltas # fix theta_0 at the first place
    centers = []
    for i in range(num_classes):
        theta_i = theta_0 + deltas[i]
        centers.append(radius * np.array([np.cos(theta_i), np.sin(theta_i)]))
    return centers

def sample_cluster_positions(num_per_class, category_list, bound, r, scale=0.05):
    """
    sample i.i.d. gaussian 2-d positions centered on 'center'
    return positions: (num_objs, 2)  [x, y]
    """
    centers = sample_centers(len(category_list))
    balls_dict = {key: [] for key in category_list}
    for i in range(len(category_list)):
        positions = np.random.normal(size=(num_per_class, 2)) * scale
        positions += centers[i]
        positions = np.clip(positions, -(bound - r), (bound - r))
        balls_dict[category_list[i]] = positions
    return balls_dict

def sample_stacking_positions(num_per_class, category_list, bound, r, scale=0.05):
    """
    sample i.i.d. gaussian 2-d positions centered on 'center'
    return positions: (num_objs, 3)  [x, y]
    """
    centers = sample_centers(len(category_list))
    balls_dict = {key: [] for key in category_list}
    for i in range(len(category_list)):
        heights = (np.arange(1, num_per_class+1, 1) * 2 - 1) * r
        xy_positions = np.zeros(num_per_class, 2)
        xy_positions += centers[i]
        positions = np.concatenate([xy_positions, heights.reshape(-1, 1)])
        balls_dict[category_list[i]] = positions
    return balls_dict

# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Random ----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def sample_random_positions(num_per_class, category_list, bound, r, pos_dim=2, scale=0.3):
    """
    return positions: (num_objs, 2)  [x, y]
    scale = self.bound / 1
    """
    balls_dict = {key: [] for key in category_list}
    for i in range(len(category_list)):
        positions = np.random.uniform(-1, 1, size=(num_per_class, pos_dim)) * scale
        balls_dict[category_list[i]] = positions
    return balls_dict

# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Circling / Circling + Clustering ----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def sample_circle_positions(num_per_class, category_list, bound, r, scale=0.05,
                         dynamic_center=True, random_rotate=True, is_permutation=True, random_color=False):
    """
    random_color = True: Circling
    random_color = False: Circling + Clustering
    """
    # center margin: the edge of circle will not be too close to the wall bound
    center_margin = 0.1
    bound *= 1 - center_margin

    # sample a radius
    num_objs = num_per_class * len(category_list)
    r_min = r / np.sin(np.pi / num_objs)
    r_max = bound - r
    assert r_max > r_min
    radius = np.random.uniform(r_min, r_max, size=(1,))

    # sample center
    cur_r = np.max(radius)
    center_bound = bound - r - cur_r
    if dynamic_center:
        center = np.random.uniform(-center_bound, center_bound, size=(2,)).reshape(-1, 2)
        center = np.repeat(center, num_objs, axis=0)
    else: # fixed center
        center = np.zeros((num_objs, 2))

    thetas = np.array(range(num_objs)) * (2 * np.pi / num_objs)
    if random_rotate:
        thetas += np.random.uniform(0, 2 * np.pi)
    if is_permutation:
        category_list = np.random.permutation(category_list)
    if random_color:
        thetas = np.random.permutation(thetas)
    balls_dict = {key: [] for key in category_list}

    for i in range(len(category_list)):
        selected_thetas = thetas[i * num_per_class: (i + 1) * num_per_class]
        positions = np.concatenate([(radius * np.cos(selected_thetas)).reshape((-1, 1)),
                                    (radius * np.sin(selected_thetas)).reshape((-1, 1))], axis=-1)
        positions += center[i * num_per_class: (i + 1) * num_per_class]
        positions = np.clip(positions, -(bound - r), (bound - r))
        balls_dict[category_list[i]] = positions

    return balls_dict


