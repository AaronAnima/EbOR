import numpy as np
import random
import itertools
from ipdb import set_trace

def get_pseudo_likelihood(state, pattern, num_per_class, category_list, bound, r):
    if pattern == 'Cluster':
        balls_dict = cluster_pseudo_likelihood(state, num_per_class, category_list, bound, r, scale=0.05)
    elif pattern == 'Circle':
        balls_dict = circle_pseudo_likelihood(state, num_per_class, category_list)
    if pattern == 'CircleCluster':
        balls_dict = circlecluster_pseudo_likelihood(state, num_per_class, category_list)
    else:
        raise NotImplementedError()
    return balls_dict

# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Clustering ----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def pdf_color(center, positions, scale=0.05):
    z = (positions.reshape(-1, 2) - center)/scale
    densities = np.exp(-z**2/2)/np.sqrt(2*np.pi)
    return densities.prod()


def get_centers_list(num_classes, radius=0.18, random_shuffle=True, random_rotate=True):
    centers_list = []
    # init sample-center for each class
    theta0s_list = []
    if random_rotate:
        for class_idx in range(num_classes):
            theta0s_list.append((class_idx/num_classes) * 2 * np.pi)
    else:
        theta0s_list.append(0)
    deltas_list = []
    if random_shuffle:
        deltas_list = list(itertools.permutations([2 * idx * np.pi / num_classes for idx in range(1, num_classes)]))
        deltas_list = [[0]+item for item in deltas_list]
    else:
        deltas_list.append([2 * idx * np.pi / num_classes for idx in range(0, num_classes)])
    for theta_0 in theta0s_list:
        for deltas in deltas_list:
            centers = []
            for i in range(num_classes):
                theta_i = theta_0 + deltas[i]
                centers.append(radius * np.array([np.cos(theta_i), np.sin(theta_i)]))
            centers_list.append(centers_list)
    return centers_list

def cluster_pseudo_likelihood(state_np, num_per_class, category_list, bound, r, radius=0.18, scale=0.05, is_normed=True):
    assert len(state_np.shape) == 1 and state_np.shape[0] == num_per_class * len(category_list) * 3
    # filt out labels
    state_np = state_np.reshape((-1, 3))[:, :2].reshape(-1)
    # scaling
    if is_normed:
        state_ = state_np*(bound - r)
    else:
        state_ = state_np
    # spliting positions by color
    clusters_by_color = [state_[2*idx*num_per_class:2*(idx+1)*num_per_class] for idx in range(len(category_list))]
    # get all possible centers
    centers_list = get_centers_list(len(category_list), radius)
    # calc pdf for GMM
    res = 0
    for centers in centers_list:
        cur_pdf = 1
        for center, positions in zip(centers, clusters_by_color):
            cur_pdf *= pdf_color(center, positions, scale=scale)
        res += cur_pdf / len(centers_list)
    return res


# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Circling  ----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def get_delta_thetas_std(thetas):
    thetas_sorted = np.sort(thetas)
    delta_thetas = thetas_sorted - np.concatenate(
        [np.array([thetas_sorted[-1] - 2 * np.pi]), thetas_sorted[0:-1]])  # deltas = [a_1+2pi - a_n, a_2 - a_1, ... a_n - a_n-1]
    theta_std = np.std(delta_thetas)
    return theta_std

def circle_pseudo_likelihood(state_np, num_per_class, category_list):
    assert len(state_np.shape) == 1 and state_np.shape[0] == num_per_class * len(category_list) * 3

    positions = state_np.reshape(-1, 3)[:, :2]
    positions_centered = positions - np.mean(positions, axis=0)
    positions_centered /= np.max(np.abs(positions_centered))
    radiuses = np.sqrt(np.sum(positions_centered**2, axis=-1))
    radius_std = np.std(radiuses)
    thetas = np.arctan2(positions_centered[:, 1], positions_centered[:, 0]) # theta = atan2(y, x)
    theta_std = get_delta_thetas_std(thetas)
    return np.exp(-(theta_std+radius_std))


# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Circling + Clustering ----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------


def get_positions_std(positions):
    mean_pos = np.mean(positions, axis=0)
    dists = np.sqrt(np.sum((positions - mean_pos)**2, axis=-1))
    return np.std(dists)


def get_thetas_std(thetas):
    positions = np.concatenate([np.cos(thetas).reshape(-1, 1), np.sin(thetas).reshape(-1, 1)], axis=-1)
    return get_positions_std(positions)

def circlecluster_pseudo_likelihood(state_np, num_per_class, category_list):
    assert len(state_np.shape) == 1 and state_np.shape[0] == num_per_class * len(category_list) * 3
    # calc circular likelihood
    circular_likelihood = circle_pseudo_likelihood(state_np, num_per_class, category_list)
    
    # calc positions_centered and thetas
    num_classes = len(category_list)
    positions = state_np.reshape(-1, 3)[:, :2]
    positions_centered = positions - np.mean(positions, axis=0) # [num_balls, 2]
    positions_centered /= np.max(np.abs(positions_centered))
    thetas = np.arctan2(positions_centered[:, 1], positions_centered[:, 0]) # theta = atan2(y, x)

    # intra-class std
    intra_class_std = 0
    for idx in range(num_classes):
        intra_class_std += get_thetas_std(thetas[idx*num_per_class:(idx+1)*num_per_class])

    # inter-class std
    class_centers = []
    for idx in range(num_classes):
        class_centers.append(np.mean(positions_centered[idx*num_per_class:(idx+1)*num_per_class], axis=0))
    inter_class_std = np.std(np.array(class_centers))

    return np.exp(inter_class_std - intra_class_std) * circular_likelihood



