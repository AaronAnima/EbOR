import numpy as np

def get_example_positions(n_per_class, category_list, patterns,  bound, r):
    if patterns == 'clustering':
        balls_dict = get_cluster_positions(n_per_class, category_list, bound, r, scale=0.05)
    elif patterns == 'circle':
        balls_dict = get_circle_positions(n_per_class, category_list, bound, r, scale=0.05, random_color=True)
    if patterns == 'circle_cluster':
        balls_dict = get_circle_positions(n_per_class, category_list, bound, r, scale=0.05, random_color=False)
    elif patterns == 'random':
        balls_dict = get_random_positions(n_per_class, category_list, bound, r, scale=0.3)
    return balls_dict

# get center positions for each class, used in clustering
def get_centers(num_classes, radius= 0.18, random_flip=True, random_rotate=True):
        """
        sample centers for each category, note that each pair of centers are equally spaced
        random_flip: determine r, g, b or r, b, g   clockwise
        random_rotate: determine direction angle of r's center(0, 120, 240)
        return red_center:[2], green_center: [2], blue_center: [2]
        """
        # init sample-center for each class
        if random_rotate:
            theta_0 = np.random.uniform(0, 2*np.pi) # random initial angle
            # theta_0 = np.random.randint(0, num_classes) * 2 * np.pi
        else:
            theta_0 = 0
        delta = [-2 * np.pi / num_classes, 2 * np.pi / num_classes]
        if random_flip:
            coin = np.random.randint(2)
        else:
            coin = 0
        centers = []
        for i in range(num_classes):
            theta_i = theta_0 + i * delta[coin]
            centers.append(radius * np.array([np.cos(theta_i), np.sin(theta_i)]))
        # # theta_green = theta_red + delta[coin]
        # # theta_blue = theta_red + delta[1 - coin]
        # radius = 0.18
        # red_center = radius * np.array([np.cos(theta_red), np.sin(theta_red)])
        # green_center = radius * np.array([np.cos(theta_green), np.sin(theta_green)])
        # blue_center = radius * np.array([np.cos(theta_blue), np.sin(theta_blue)])
        return centers

def get_cluster_positions(n_per_class, category_list, bound, r, scale=0.05):
        """
        sample i.i.d. gaussian 2-d positions centered on 'center'
        return positions: (n_boxes, 3)  [x, y, class_name]
        """
        centers = get_centers(len(category_list))
        balls_dict = {key: [] for key in category_list}
        for i in range(len(category_list)):
            positions = np.random.normal(size=(n_per_class, 2)) * scale
            positions += centers[i]
            positions = np.clip(positions, -(bound - r), (bound - r))
            balls_dict[category_list[i]] = positions
        return balls_dict

def get_random_positions(n_per_class, category_list, bound, r, scale=0.3):
        """
        sample i.i.d. gaussian 2-d positions centered on 'center'
        return positions: (n_boxes, 3)  [x, y, class_name]
        scale = self.bound / 1
        """
        balls_dict = {key: [] for key in category_list}
        for i in range(len(category_list)):
            positions = np.random.uniform(-1, 1, size=(n_per_class, 2)) * scale
            balls_dict[category_list[i]] = positions
        return balls_dict


def get_circle_positions(n_per_class, category_list, bound, r, scale=0.05,
                         dynamic_center=True, random_rotate=True, is_permutation=True, random_color=False):
    # sample a radius
    n_boxes = n_per_class * len(category_list)
    r_min = r / np.sin(np.pi / n_boxes)
    r_max = bound - r
    assert r_max > r_min
    radius = np.random.uniform(r_min, r_max, size=(1,))

    # get center
    cur_r = np.max(radius)
    center_bound = bound - r - cur_r
    if dynamic_center:
        center = np.random.uniform(-center_bound, center_bound, size=(2,)).reshape(-1, 2)
        center = np.repeat(center, n_boxes, axis=0)
    else: # fixed center
        center = np.zeros((n_boxes, 2))

    thetas = np.array(range(n_boxes)) * (2 * np.pi / n_boxes)
    if random_rotate:
        thetas += np.random.uniform(0, 2 * np.pi)
    if is_permutation:
        category_list = np.random.permutation(category_list)
    if random_color:
        thetas = np.random.permutation(thetas)
    balls_dict = {key: [] for key in category_list}

    for i in range(len(category_list)):
        selected_thetas = thetas[i * n_per_class: (i + 1) * n_per_class]
        positions = np.concatenate([(radius * np.cos(selected_thetas)).reshape((-1, 1)),
                                    (radius * np.sin(selected_thetas)).reshape((-1, 1))], axis=-1)
        positions += center[i * n_per_class: (i + 1) * n_per_class]
        positions = np.clip(positions, -(bound - r), (bound - r))
        balls_dict[category_list[i]] = positions
    return balls_dict
