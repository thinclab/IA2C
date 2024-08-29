# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
utils.py

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import numpy as np
import scipy

from collections import namedtuple
from PIL import Image, ImageDraw

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


def dict2namedtuple(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def pos_to_ang(pos):
    """
    CCW
    Convert x,y position to angle in radians
    Range: [0, 2PI]

    Args:
        pos (np.array): x,y position array
    """
    degrees = np.degrees(np.arctan2(pos[0], pos[1]))
    if degrees > 90.0:
        degrees = 450.0 - degrees
    else:
        degrees = 90.0 - degrees
    return np.radians(degrees)


def pos_to_angle(pos):
    """
    CCW
    Convert x,y position to angle in radians

    Args:
        pos (np.array): x,y position array
    """
    angle = np.arctan(pos[1] / pos[0])
    # add to attacker angle based on what quadrant attacker is in:
    # 1:0, 2:pi, 3:pi, 4:2pi
    if pos[0] < 0:
        angle += np.pi
    elif pos[1] < 0:
        angle += 2 * np.pi
    return angle


def angle_to_pos(angle, radius):
    """
    CCW
    Convert angle in radians to x,y postion

    Args:
        angle (float): Angle in radians
        radius (float): Distance from center
    Returns:
        pos (np.array): x,y position array
    """
    pos = np.array([np.cos(angle) * radius, np.sin(angle) * radius])

    return pos


def calculate_shortest_cartesian_directon(angle_a, angle_b, radius, center=None):
    """
    Determines direction of the shortest arc traveled between angle a and angle b

    https://math.stackexchange.com/questions/1525961/determine-direction-of-an-arc-cw-ccw

    Args:
        angle_a (float):
        angle_b (float):
        radius (float):
        center (np.array):

    Returns:
        Direction scalar
    """
    pos_a = angle_to_pos(angle_a, radius)
    pos_b = angle_to_pos(angle_b, radius)

    if center is None:
        distance = ((pos_a[0]) * (pos_b[1])) - ((pos_a[1]) * (pos_b[0]))
    else:
        distance = ((pos_a[0] - center[0]) * (pos_b[1] - center[1])) - \
                    ((pos_a[1] - center[1]) * (pos_b[0] - center[0]))

    if distance > 0.0:
        # CCW
        direction = 1
    elif distance < 0.0:
        # CW
        direction = -1
    else:
        # Either direction is valid if direction is equal to 0
        direction = np.random.choice([1, -1])

    return direction, distance


def calculate_shortest_polar_directon(angle_a, angle_b):
    """
    Determines direction of the shortest arc traveled between angle a and angle b

    Args:
        angle_a (float):
        angle_b (float):

    Returns:
        Direction scalar
    """
    distance = (np.mod(((angle_b - angle_a) + np.pi), (2 * np.pi)) - np.pi)

    if distance > 0.0:
        # CCW
        direction = 1
    elif distance < 0.0:
        # CW
        direction = -1
    else:
        # Either direction is valid if direction is equal to 0
        direction = np.random.choice([1, -1])

    # return direction, np.rad2deg(distance)
    return direction, distance


###########################################
#         Adjacency Matrix Ops            #
###########################################
def tf_normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    import tensorflow as tf
    d_inv_sqrt = tf.pow(tf.reduce_sum(adj, 1), -0.5)
    mask = tf.math.is_inf(d_inv_sqrt)
    indices = tf.where(mask)
    updates = tf.zeros(len(indices), dtype=tf.dtypes.float64)
    d_inv_sqrt = tf.tensor_scatter_nd_update(d_inv_sqrt, indices, updates)
    d_mat_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
    return tf.linalg.matmul(tf.linalg.matmul(adj, d_mat_inv_sqrt), d_mat_inv_sqrt, transpose_a=True)


def torch_normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    import torch
    d_inv_sqrt = adj.sum(1).pow(-0.5)
    torch.nan_to_num(d_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0, out=d_inv_sqrt)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return adj.matmul(d_mat_inv_sqrt).transpose(1,0).matmul(d_mat_inv_sqrt)


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    adj = scipy.sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    import torch
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_tf_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a tf sparse tensor.
    """
    import tensorflow as tf
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = list(map(list, zip(sparse_mx.row, sparse_mx.col)))
    indices.sort()
    values = sparse_mx.data.tolist()
    shape = list(sparse_mx.shape)
    return tf.sparse.SparseTensor(indices, values, shape)


###########################################
#            Grid Draw Functions          #
###########################################
def draw_grid(rows, cols, cell_size=50, fill='black', line_color='black'):
    height = rows * cell_size
    width = cols * cell_size
    image = Image.new(mode='RGB', size=(width, height), color=fill)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = cell_size

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=line_color)

    x = image.width - 1
    line = ((x, y_start), (x, y_end))
    draw.line(line, fill=line_color)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=line_color)

    y = image.height - 1
    line = ((x_start, y), (x_end, y))
    draw.line(line, fill=line_color)

    del draw

    return image


def fill_cell(image, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    col, row = pos
    row, col = row * cell_size, col * cell_size
    margin *= cell_size
    x, y, x_dash, y_dash = row + margin, col + margin, row + cell_size - margin, col + cell_size - margin
    ImageDraw.Draw(image).rectangle([(x, y), (x_dash, y_dash)], fill=fill)


def write_cell_text(image, text, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    col, row = pos
    row, col = row * cell_size, col * cell_size
    margin *= cell_size
    x, y = row + margin, col + margin
    ImageDraw.Draw(image).text((x, y), text=text, fill=fill)


def draw_cell_outline(image, pos, cell_size=50, fill='black'):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    ImageDraw.Draw(image).rectangle([(row, col), (row + cell_size, col + cell_size)], outline=fill, width=3)


def draw_circle(image, pos, cell_size=50, fill='black', radius=0.3):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * radius
    x, y = row + gap, col + gap
    x_dash, y_dash = row + cell_size - gap, col + cell_size - gap
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], outline=fill, fill=fill)


def draw_border(image, border_width=1, fill='black'):
    width, height = image.size
    new_im = Image.new("RGB", size=(width + 2 * border_width, height + 2 * border_width), color=fill)
    new_im.paste(image, (border_width, border_width))
    return new_im


def draw_score_board(image, score, board_height=30):
    im_width, im_height = image.size
    new_im = Image.new("RGB", size=(im_width, im_height + board_height), color='#e1e4e8')
    new_im.paste(image, (0, board_height))

    _text = ', '.join([str(round(x, 2)) for x in score])
    ImageDraw.Draw(new_im).text((10, board_height // 3), text=_text, fill='black')
    return new_im


def draw_attention_circle(image, pos, cell_size=50, fill='white', outline='black', radius=0.3, width=1):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * radius
    x, y = row + gap, col + gap
    x_dash, y_dash = row + cell_size - gap, col + cell_size - gap
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], fill=fill, outline=outline, width=width)
