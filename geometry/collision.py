import numpy as np


def create_box(size=0.1):
    box = np.array([[-size/2, -size/2, 0, 1],
                    [-size/2, size/2, 0, 1],
                    [size/2, size/2, 0, 1],
                    [size/2, -size/2, 0, 1]])
    return box


def check_collision(box1, T1, box2, T2):
    box1_world = box1 @ T1.T
    box2_world = box2 @ T2.T

    x1_min = np.min(box1_world[:, 0])
    x1_max = np.max(box1_world[:, 0])
    y1_min = np.min(box1_world[:, 1])
    y1_max = np.max(box1_world[:, 1])

    x2_min = np.min(box2_world[:, 0])
    x2_max = np.max(box2_world[:, 0])
    y2_min = np.min(box2_world[:, 1])
    y2_max = np.max(box2_world[:, 1])

    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)
