4# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
alternate_policies.py

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from numpy import sinh, sqrt
from scipy.optimize import newton, newton_krylov, brentq  # James
from scipy.spatial import Delaunay

from multiagent_particle_env.utils import pos_to_ang, pos_to_angle, calculate_shortest_polar_directon


__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


###########################################
#                 Globals                 #
###########################################
# variables used for demo "corners" policy
step = 0
target_x = 0
target_y = 0

###########################################
#               Utilities                 #
###########################################
def current_milli_time():
    return round(time.time() * 1000)


###########################################
#         Meta-Reasoning Policies         #
###########################################

def prey_analysis(agent, world):

    """
    Beginning outline for an analysis of prey movements and velocity to calculate the most energy efficient way of
    capturing a prey (currently unimplemented)

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks
    """

    raise NotImplementedError(
        "This function represents an algorithm that may later be implemented in a future metareasoning algorithm.")

    # Collect prey agents
    prey = []
    for other in world.policy_agents:
        if not other.adversary:
            prey.append(other)

    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    my_prey = prey[0]
    prey_x = my_prey.state.p_pos[0]
    prey_y = my_prey.state.p_pos[1]
    prey_speed = my_prey.max_speed


def metareasoning_switch_policy_outline(agent, world):
    """
    Beginning outline for a future metareasoning algorithm (currently unimplemented)

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks
    """

    raise NotImplementedError("This function represents an outline of a future metareasoning algorithm.")

    # state variable
    state = "delaunay"

    if state == "delaunay":
        delaunay_triangulation_prey_fixed(agent, world)
    elif state == "distance-minimize":
        distance_minimizing_fixed_strategy(agent, world)
    elif state == "trapper":
        trapper_fixed_strategy(agent, world)


###########################################
#         Predator-Prey Policies          #
#                 Prey                    #
###########################################

def utility_circumcenter(triangle):
    """
    Returns the circumcenter of a triangle

    Args:
        triangle (numpy.ndarray): an array of 3 points

    Returns:
        circumcenter (list): x and y coordinates of the triangle's circumcenter
    """
    a = triangle[0]
    b = triangle[1]
    c = triangle[2]
    ax = float(a[0])
    ay = float(a[1])
    bx = float(b[0])
    by = float(b[1])
    cx = float(c[0])
    cy = float(c[1])
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d

    circumcenter = [ux, uy]
    return circumcenter


def delaunay_triangulation_prey_fixed(agent, world):
    """
    Fixed evader strategy to use dulanay triangulation to maximize distance from pursuer

    CURRENTLY OPTIMIZED FOR ONE PURSUER AND ONE EVADER

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Collect prey agents
    pursuers = []
    for other in world.policy_agents:
        if other.adversary:
            pursuers.append(other)

    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    my_pursuer = pursuers[0]
    pursuer_x = my_pursuer.state.p_pos[0]
    pursuer_y = my_pursuer.state.p_pos[1]

    # boundaries and pursuer position
    north_east = [1, 1]
    south_east = [1, -1]
    south_west = [-1, -1]
    north_west = [-1, 1]
    origin = [0, 0]
    north = [0, 1]
    south = [0, -1]
    east = [1, 0]
    west = [-1, 0]

    pursuer_position = [pursuer_x, pursuer_y]
    my_position = [agent.state.p_pos[0], agent.state.p_pos[0]]

    dist_prey_to_pursuer = math.sqrt(((my_position[0] - pursuer_position[0]) ** 2)
                            + ((my_position[1] - pursuer_position[1]) ** 2))

    # triangulation points
    points = np.array([pursuer_position, north_east, south_east, south_west, north_west])

    delaunay_triangulation = Delaunay(points)

    # arrays for calculated circumcenters and radii
    circumcenters = []
    radii = []

    for triangle in points[delaunay_triangulation.simplices]:
        # calculate circumcenter
        cc = utility_circumcenter(triangle)

        # check if cc lies inside perimeter
        if cc[0] >= -1 and cc[0] <= 1 and cc[1] >= -1 and cc[1] <= 1:
            circumcenters.append(cc)

            # calculate cc distance from pursuer
            radius = math.sqrt((cc[0] - pursuer_x)**2 + (cc[1] - pursuer_y)**2)
            radii.append(radius)

    # calculate highest radius and index
    highest_radius = max(radii)
    highest_radius_index = radii.index(highest_radius)

    # assign target x and y coords
    target_x = circumcenters[highest_radius_index][0]
    target_y = circumcenters[highest_radius_index][1]

    x_n, y_n = target_x / np.linalg.norm(np.array([target_x, target_y])), target_y / np.linalg.norm(
        np.array([target_x, target_y]))

    # assigns actions
    agent.action.u[0] = x_n
    agent.action.u[1] = y_n

    # Scale action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


###########################################
#         Predator-Prey Policies          #
#               Predators                 #
###########################################

def distance_minimizing_fixed_strategy(agent, world):
    """
    Distance-minimizing fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Collect prey agents
    prey = []
    for other in world.policy_agents:
        if not other.adversary:
            prey.append(other)

    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    my_prey = prey[0]
    x = my_prey.state.p_pos[0] - agent.state.p_pos[0]
    y = my_prey.state.p_pos[1] - agent.state.p_pos[1]

    x_n, y_n = x / np.linalg.norm(np.array([x, y])), y / np.linalg.norm(np.array([x, y]))

    agent.action.u[0] = x_n
    agent.action.u[1] = y_n

    # Scale action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


def random_fixed_strategy(agent, world):
    """
    Random fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    # Set random action
    random_act = np.random.random()*2*np.pi
    agent.action.u[0] = np.cos(random_act)
    agent.action.u[1] = np.sin(random_act)

    # Scale random action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


def spring_fixed_strategy(agent, world):
    """
    Random fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    k = 10
    F = np.zeros((2,))

    for other in world.agents:
        dis = np.linalg.norm((other.state.p_pos - agent.state.p_pos))
        if other.adversary and other != agent:
            # F += k * (np.linalg.norm(other.state.p_pos - agent.state.p_pos) - 0.5) * \
            #      (other.state.p_pos - agent.state.p_pos)
            F += k * (dis - 0.5) * ((other.state.p_pos - agent.state.p_pos) / dis)
        if not other.adversary and other != agent:
            # F += 0.4 * k * 1 / (np.linalg.norm(other.state.p_pos - agent.state.p_pos)) * \
            #      (other.state.p_pos - agent.state.p_pos)
            F += 0.2 * k * (1 / dis) * (other.state.p_pos - agent.state.p_pos)

    F = F / np.linalg.norm(F)

    # Scale spring action by acceleration
    agent.action.u = agent.accel * F

    return agent.action


def spring_fixed_strategy_2(agent, world):
    """
    Random fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    k = 10
    F = np.zeros((2,))

    for other in world.agents:
        dis = np.linalg.norm((other.state.p_pos - agent.state.p_pos))
        if other.adversary and other != agent:
            # F += k * (np.linalg.norm(other.state.p_pos - agent.state.p_pos) - 0.5) * \
            #      (other.state.p_pos - agent.state.p_pos)
            F += k * (dis - 0.5) * ((other.state.p_pos - agent.state.p_pos) / dis)
        # if not other.adversary and other != agent:
            # F += 0.4 * k * 1 / (np.linalg.norm(other.state.p_pos - agent.state.p_pos)) * \
            #      (other.state.p_pos - agent.state.p_pos)
            # F += 0.2 * k * (1 / dis) * (other.state.p_pos - agent.state.p_pos)

    F = F / np.linalg.norm(F)

    # Scale spring action by acceleration
    agent.action.u = agent.accel * F

    return agent.action


def sheep_fixed_strategy(agent, world):
    """
    Sheep fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    threat_rad = 0.5
    pred_threats = []
    for other in world.policy_agents:
        if other.adversary:
            vec = np.zeros(world.dimension_position)
            vec[0] = other.state.p_pos[0] - agent.state.p_pos[0]
            vec[1] = other.state.p_pos[1] - agent.state.p_pos[1]
            d = np.linalg.norm(vec)
            if d <= threat_rad:
                pred_threats.append(vec)

    if len(pred_threats) > 0:
        agent.action.u = -1 * (np.sum(pred_threats, 0)) * (1 / np.linalg.norm(np.sum(pred_threats, 0)))
    else:
        agent.action.u = np.zeros(world.dimension_position)
        a = np.random.random() * 2 * np.pi
        agent.action.u[0] = np.cos(a)
        agent.action.u[1] = np.sin(a)

        # Scale sheep action by acceleration
        agent.action.u = agent.accel * agent.action.u

    return agent.action


def evader_fixed_strategy(agent, world):
    """
    Evader distance-minimizing fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Collect predator agents
    preds = []
    for i, other in enumerate(world.policy_agents):
        if other.adversary:
            preds.append(other)

    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    x = []
    y = []
    for pred in preds:
        xpred = pred.state.p_pos[0] - agent.state.p_pos[0]
        ypred = pred.state.p_pos[1] - agent.state.p_pos[1]

        scale = np.linalg.norm(np.array([xpred, ypred]))
        scale = np.exp(-4 * scale)

        x.append(scale * xpred)
        y.append(scale * ypred)

    for i, obs in enumerate(world.landmarks):
        xobs = obs.state.p_pos[0] - agent.state.p_pos[0]
        yobs = obs.state.p_pos[1] - agent.state.p_pos[1]

        scale = np.linalg.norm(np.array([xobs, yobs]))

        # scale = np.max([np.exp(-2 * scale), 1e-1])
        scale = np.exp(-10 * (scale - 3))
        # print("Scale: {}".format(scale))

        xobs, yobs = xobs / np.linalg.norm(np.array([xobs, yobs])), yobs / np.linalg.norm(np.array([xobs, yobs]))

        x.append(scale * xobs)
        y.append(scale * yobs)

    x, y = np.mean(np.array([x, y]), 1)

    x_n, y_n = x / np.linalg.norm(np.array([x, y])), y / np.linalg.norm(np.array([x, y]))

    agent.action.u[0] = -x_n
    agent.action.u[1] = -y_n

    # Scale action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


def trapper_fixed_strategy(agent, world):
    """
    Trapper distance-minimizing fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Collect prey agents
    prey = []
    for i, other in enumerate(world.policy_agents):
        if not other.adversary:
            prey.append(other)

    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    my_prey = prey[0]
    xp = my_prey.state.p_pos[0] - agent.state.p_pos[0]
    yp = my_prey.state.p_pos[1] - agent.state.p_pos[1]
    r = 1.0 / 3.0

    # world.landmarks[0].state.p_pos = agent.state.p_pos

    if np.linalg.norm(np.array([xp, yp])) < r:
        if agent.counter < 10:
            x, y = xp, yp
            agent.counter += 1
        else:
            x, y = np.array([0.0, 0.0]) - agent.state.p_pos
    else:
        x, y = np.array([0.0, 0.0]) - agent.state.p_pos
        if np.linalg.norm(np.array([x, y])) < r:
            agent.counter = 0
        if np.linalg.norm(np.array([x, y])) < agent.size:
            x, y = 0.0, 0.0

    if np.linalg.norm(np.array([x, y])) != 0.0:
        x_n, y_n = x / np.linalg.norm(np.array([x, y])), y / np.linalg.norm(np.array([x, y]))
    else:
        x_n, y_n = 0.0, 0.0

    agent.action.u[0] = x_n
    agent.action.u[1] = y_n

    # Scale action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


def _double_pendulum(state):
    """
    Differential dynamics of a double pendulum

    Args:
        state (multiagent_particle_env.core.AgentState.state): State element from Agent state object
              or
              (list) Agent state 4-element list

    Returns:
        Updated agent state causing agent to act as a double pendulum
    """
    L1, L2 = 0.49, 0.49
    G = 9.8
    M1, M2 = 1.0, 1.0

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(del_) * np.cos(del_)
    dydx[1] = (M2 * L1 * state[1] * state[1] * np.sin(del_) * np.cos(del_) +
               M2 * G * np.sin(state[2]) * np.cos(del_) +
               M2 * L2 * state[3] * state[3] * np.sin(del_) -
               (M1 + M2) * G * np.sin(state[0])) / den1

    dydx[2] = state[3]

    den2 = (L2 / L1) * den1
    dydx[3] = (-M2 * L2 * state[3] * state[3] * np.sin(del_) * np.cos(del_) +
               (M1 + M2) * G * np.sin(state[0]) * np.cos(del_) -
               (M1 + M2) * L1 * state[1] * state[1] * np.sin(del_) -
               (M1 + M2) * G * np.sin(state[2])) / den2

    return dydx


def double_pendulum_perturbation_strategy(agent, world):
    """
    Double Pendulum Strategy for perturbing an agent's policy

    Updates agent's state and zero's out action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    agent.action.u = np.zeros(world.dimension_position)
    agent.state.state = state

    return agent.action


def double_pendulum_perturbation_strategy_2(agent, world):
    """
    Double Pendulum Strategy for perturbing an agent's policy.

    Updates agent's state and action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    agent.action.u = np.array([L1 * np.cos(d_state[1]), L1 * np.sin(d_state[1])]) + \
                     np.array([L2 * np.cos(d_state[3]), L2 * np.sin(d_state[3])])
    agent.state.state = state

    return agent.action


def double_pendulum_perturbation_strategy_3(agent, world):
    """
    Double Pendulum Strategy for perturbing an agent's policy.

    Updates agent's state and action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    agent.action.u = a1 + a2
    agent.state.state = state

    return agent.action


def double_pendulum_perturbation_strategy_4(agent, world):
    """
    Double Pendulum Strategy for perturbing an agent's policy

    Updates agent's state and zero's out action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    x = agent.state.p_pos[0]
    y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        if (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2) >= 1.0:
            th2 = 0.0
        elif (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2) <= -1.0:
            th2 = np.pi
        else:
            th2 = np.arccos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2))

        th1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
        w1, w2 = np.random.random(1)[0]**2, np.random.random(1)[0]**2
        state = np.array([th1, w1, th2, w2])

    if all(state == 0.0):
        if (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2) >= 1.0:
            th2 = 0.0
        elif (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2) <= -1.0:
            th2 = np.pi
        else:
            th2 = np.arccos((x**2 + y**2 - L1**2 - L2**2)/(2*L1*L2))

        th1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
        w1, w2 = np.random.random(1)[0]**2, np.random.random(1)[0]**2
        state = np.array([th1, w1, th2, w2])

    # Scaled double pendulum state
    d_state = _double_pendulum(state)/40.
    state = state + d_state

    new_x = L2*np.sin(state[2]) + L1*np.sin(state[0])
    new_y = -L2*np.cos(state[2]) - L1*np.cos(state[0])

    agent.state.p_pos[0] = new_x
    agent.state.p_pos[1] = new_y

    agent.state.state = state
    agent.action.u = np.zeros(world.dimension_position)

    agent.action.u[0] = d_state[1]
    agent.action.u[1] = d_state[3]

    return agent.action


def perturbation_strategy_old(agent, world):
    """
    Strategy for perturbing an agent's policy

    Updates agent's state and zero's out action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        # agent.state.p_pos = 0.49 * np.array([np.sin(th[0]),
        #                                      -np.cos(th[0])]) + 0.45 * np.array([np.sin(th[1]), -np.cos(th[1])])
        # agent.state.p_vel = 0.49 * np.array([np.cos(w[0]),
        #                                      -np.cos(w[0])]) + 0.45 * np.array([np.cos(w[1]), -np.cos(w[1])])
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    # world.landmarks[-1].state.p_pos = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    agent.action.u = np.zeros(world.dimension_position)
    agent.state.state = state

    return agent.action


def perturbation_strategy_2_old(agent, world):
    """
    Strategy for perturbing an agent's policy

    Updates agent's state and action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        # agent.state.p_pos = 0.49 * np.array([np.sin(th[0]),
        #                                      -np.cos(th[0])]) + 0.45 * np.array([np.sin(th[1]), -np.cos(th[1])])
        # agent.state.p_vel = 0.49 * np.array([np.cos(w[0]),
        #                                      -np.cos(w[0])]) + 0.45 * np.array([np.cos(w[1]), -np.cos(w[1])])
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    # world.landmarks[-1].state.p_pos = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    agent.action.u = np.array([L1 * np.cos(d_state[1]), L1 * np.sin(d_state[1])]) + \
                     np.array([L2 * np.cos(d_state[3]), L2 * np.sin(d_state[3])])
    agent.state.state = state

    return agent.action


def perturbation_strategy_3_old(agent, world):
    """
    Strategy for perturbing an agent's policy

    Updates agent's state and action

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # x = agent.state.p_pos[0]
    # y = agent.state.p_pos[1]
    L1, L2 = 0.49, 0.49

    try:
        state = agent.state.state
    except:
        th = np.random.uniform(-np.pi, np.pi, world.dimension_position)
        w = np.zeros(world.dimension_position)
        # w = np.random.uniform(-0.2, 0.2, world.dimension_position)
        # agent.state.p_pos = 0.49 * np.array([np.sin(th[0]),
        #                                      -np.cos(th[0])]) + 0.45 * np.array([np.sin(th[1]), -np.cos(th[1])])
        # agent.state.p_vel = 0.49 * np.array([np.cos(w[0]),
        #                                      -np.cos(w[0])]) + 0.45 * np.array([np.cos(w[1]), -np.cos(w[1])])
        state = [th[0], w[0], th[1], w[1]]

    # Scaled double pendulum state
    d_state = _double_pendulum(state) / 40.0
    state = state + d_state

    p1 = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    p2 = np.array([L2 * np.sin(state[2]), -L2 * np.cos(state[2])])
    v1 = np.array([L1 * np.sin(state[1]), -L1 * np.cos(state[1])])
    v2 = np.array([L2 * np.sin(state[3]), -L2 * np.cos(state[3])])
    a1 = np.array([L1 * np.sin(d_state[1]), -L1 * np.cos(d_state[1])])
    a2 = np.array([L2 * np.sin(d_state[3]), -L2 * np.cos(d_state[3])])

    agent.state.p_pos = p1 + p2
    agent.state.p_vel = v1 + v2

    # world.landmarks[-1].state.p_pos = np.array([L1 * np.sin(state[0]), -L1 * np.cos(state[0])])
    agent.action.u = a1 + a2
    agent.state.state = state

    return agent.action


#counter = 0 #for debugging

def bearing_strategy(agent, world):
    """
    Bearing-angle fixed strategy for changing an agent's policy

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    debug = False  # Set to True for debugging, plots
    if debug: global counter #For debugging plots
    def toDeg(x):  # For convenience in debugging
        return (x * 180 / math.pi)    

    # Collect predatory and prey agents (prey borrowed from distance_minimizing_fixed_strategy)
    prey = []
    for other in world.policy_agents:
        if not other.adversary:
            prey.append(other)

    my_prey = prey[0]  # because there might be more than one, but then you ignore the rest?? Pick the closest one?

    original_action = copy.deepcopy(agent.action)  # In case I need to return unchanged as filler, shouldn't be needed for final version, but in practice, I seem to need it for rounding reasons
    agent.action.u = np.zeros(world.dimension_position)  # world.dimension_position = 2, not sure why this is necessary

    # Now come some weird names that help me with testing
    # Pursuer
    x_p = agent.state.p_pos[0]
    y_p = agent.state.p_pos[1]
    # agent_max_speed = 1
    V_p = agent.max_speed  # Max speed
    V_P = agent.max_speed  # Actual speed. Assume max speed, not actual speed from agent.stat.p_vel Might need to check that not None, not sure this needs to be normed
    # Evader
    x_e = my_prey.state.p_pos[0]
    y_e = my_prey.state.p_pos[1]
    V_E = np.linalg.norm(my_prey.state.p_vel)  # Actual speed. Assuming p_vel is change in [x,y], new position is p_pos + p_vel * dt (where dt is timestep, .1)
    # Setting the stage
    r = math.sqrt((x_p - x_e) ** 2 + (y_p - y_e) ** 2)  # Distance between pursuer and evader
    if r == 0:  # Capture
        # Plot for testing
        if debug:
            plt.plot([x_p], [y_p], 'bo')  # pursuer
            plt.plot([x_e], [y_e], 'ro')  # evader
            plt.plot([x_e + my_prey.state.p_vel[0]], [y_e + my_prey.state.p_vel[1]], 'r*')
            plt.plot([x_p, x_e], [y_p, y_e], 'black')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axhline(y=0, color='gray')
            plt.axvline(x=0, color='gray')
            plt.title("r=0")
            plt.grid()
            plt.show()
        return (original_action)
    if (y_e - y_p < 1): r *= -1  # This might make calculating intersection easier?
    nu = V_E / V_P  # Mike said to use actual speeds

    # Mike's math magic to calculate unambiguous phi
    r_E_P_3D = np.array([x_p - x_e, y_p - y_e, 0])  # Vector from evader to pursuer, 3D
    V_E_3D = np.array([my_prey.state.p_vel[0], my_prey.state.p_vel[1], 0])
    temp = np.cross(r_E_P_3D, V_E_3D)
    phi = math.atan2(temp[2], np.dot(r_E_P_3D[0:2], V_E_3D[0:2]))#phi is angle between r and pursuer's velocity

    vers = ""

    if nu >= 1:  # nu=V_E/V_P, faster evader
        phi_star = math.asin(1 / nu)
    else:  # Faster pursuer
        phi_star = phi  # To avoid div by zero error. phi_star should be irrelevant when V_E<V_p anyway.

    if V_E < V_p or abs(phi) <= phi_star:  # Mike had these separate
        vers = "Capture possible"
        theta = -math.asin((V_E / V_P) * math.sin(phi))  # Equation 4, angle between r and evader's velocity
    else:
        vers = "Capture impossible"
        def no_capture_theta(theta, phi=phi, V_P=V_P, V_E=V_E):
            return (math.sqrt(1 - V_P ** 2 / V_E ** 2) * math.sin(theta) - math.sin(theta + phi))
        if phi >= 0:
            theta = -brentq(no_capture_theta, 0, math.pi, maxiter=500)
        else:
            theta = -brentq(no_capture_theta, -math.pi, 0, maxiter=500)

    rho = math.pi - abs(theta) - abs(phi)  # angle opposite r
    if rho == 0:  # r=0, already collided, or rounding makes it look like they did
        print("Rho = 0, this shouldn't happen")
        return(original_action)#Perhaps not best action
    elif rho == math.pi:#will happen if evader velocity is 0
        p = abs(r) #distance between pursuer and intersection = distance between pursuer and evader
        if debug:
            print("Evader velocity effectively 0, ", my_prey.state.p_vel)
            print("distance between pursuer and intersection = distance between pursuer and evader")
    else:
        p = abs((r / math.sin(rho)) * math.sin(phi))  # distance between pursuer and intersection

    m_r = (y_p - y_e) / (x_p - x_e)
    # Find slope of p using theta and difference between theta and x axis
    if m_r != None:
        m_p = math.tan(math.atan(m_r) + theta)  # Calculate slope of p using theta plus angle between theta and x axis
    else:
        m_p = math.pi / 2 - math.tan(math.atan(0) + theta)  # Double check, I don't remember what exactly this is doing

    # OK, where do p and e meet?
    x_change = math.sqrt(abs(p ** 2 / (1 + m_p ** 2)))# We'll change sign later if necessary
    x_c = x_p + x_change  # So the location of the intersection is the location of the pursuer + the x and y changes
    y_change = m_p * x_change
    y_c = y_p + y_change
    # [x_c,y_c]#This is the point where pursuer and evader would meet

    # Use change from pursuer to find new pursuer velocity (if p were of length V_p, what would it's x and y components be?)
    x_change_scaled = V_P * (1 / p) * x_change  # It's V_P/sqrt(1+m_p**2), but it should be same sign as x_change, so I'm using a roundabout way
    y_change_scaled = m_p * x_change_scaled
    new_vel = np.array([x_change_scaled, y_change_scaled])  # The pursuer's new velocity is along the slope of p, scaled by speed

    tried_sign = False
    sf = 1  # -1 to change the direction along m_p that pursuer will go (due to sqrt sign ambiguity)
    while True:
        # print("Looping...")
        A = np.array([x_e + my_prey.state.p_vel[0], y_e + my_prey.state.p_vel[1]])#Evader future location (might be problem if overrunning prey at max speed)
        B = np.array([x_e, y_e])#Evader location
        C = np.array([x_c, y_c])#Intersection
        # If points are not on the same line (and so their area isn't 0), either we chose the wrong intersection angle (should be impossible now) or we picked the wrong sign for sqrt when calculating pursuer's x change
        # Try both separate and together, they should ultimately be on same line
        bad_area = abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1])) > 0.00001  # Should be 0
        bad_order = abs(np.linalg.norm(C - A) - np.linalg.norm(A - B) - np.linalg.norm(C - B)) < 0.00001  # Should be greater than 0 (unless stationary prey!)
        if debug: print("bad_area: ", bad_area, ", bad_order: ", bad_order )
        if abs(np.linalg.norm(B - A))<.0001:#if evader is stationary, use chaser code, this could be moved earlier to be slightly more efficient
           x = my_prey.state.p_pos[0] - agent.state.p_pos[0]
           y = my_prey.state.p_pos[1] - agent.state.p_pos[1]
        
           x_n, y_n = x / np.linalg.norm(np.array([x, y])), y / np.linalg.norm(np.array([x, y]))
        
           agent.action.u[0] = x_n
           agent.action.u[1] = y_n
        
           # Scale action by acceleration
           agent.action.u = agent.accel * agent.action.u
           
           #Add plotting
           if debug:
               plt.plot([x_p], [y_p], 'bo', alpha=.7)  # pursuer
               plt.plot([x_p, x_e], [y_p, y_e], 'b--')
               plt.plot([x_p + x_n], [y_p + y_n], 'b*')
               plt.plot([x_e], [y_e], 'ro', alpha=.7)  # evader
               #plt.plot([x_e, x_c], [y_e, y_c], 'r')
               plt.plot([x_e + my_prey.state.p_vel[0]], [y_e + my_prey.state.p_vel[1]], 'm*')
               #plt.plot([x_p, x_e], [y_p, y_e], 'black')
               #plt.plot([x_c], [y_c], 'gx')  # intersection
               plt.plot([x_p + agent.action.u[0]], [y_p + agent.action.u[1]], 'y*')  # Action?
               plt.title(vers + " bad_area:" + str(bad_area)+ " bad_order:" + str(bad_order) + " stat:" + str(abs(np.linalg.norm(B - A))<.00001) + " sf=" +str(sf) + '\nphi=' + str(math.floor(toDeg(phi))) + ' theta=' + str(math.floor(toDeg(theta))) + " \nEvel: "+str(my_prey.state.p_vel) + " \nPvel: "+str([x_n,y_n]))
               plt.gca().set_aspect('equal', adjustable='box')
               plt.axhline(y=0, color='gray')
               plt.axvline(x=0, color='gray')
               plt.xlim(-1, 1)
               plt.ylim(-1, 1)
               plt.grid()
               # plt.show()
               plt.savefig('C:/Users/erin.g.zaroukian/Desktop/Multi-Agent-Behaviors-Team/testPlots/' + str(counter) + "-" + str(counter%3) + '.png') 
               plt.clf()
               
               counter += 1#For debugging plots
           
           return agent.action
           
        elif not tried_sign and (bad_area or bad_order):
            sf *= -1
            tried_sign = True
            if debug: print("Flipping...")
        else:
            if debug: print("Fixed/OK! - bad_area: ", bad_area, ", bad_order: ", bad_order )
            break
        # Redo some calculations - OK, where do p and e meet?
        # Calculate the intersection's offset from pursuer
        x_change = sf * math.sqrt(abs(p ** 2 / (1 + m_p ** 2)))  # We'll change sign later if necessary
        x_c = x_p + x_change  # So the location of the intersection is the location of the pursuer + the x and y changes
        y_change = m_p * x_change
        y_c = y_p + y_change
        # [x_c,y_c]#This is the point where pursuer and evader would meet
            
        # Use change from pursuer to find new pursuer velocity (if p were of length V_p, what would it's x and y components be?)
        x_change_scaled = V_P * (1 / p) * x_change  # It'sV_P/(1+m_p**2), but it should be same sign as x_change, so I'm using a roundabout way
        y_change_scaled = m_p * x_change_scaled
        new_vel = np.array([x_change_scaled, y_change_scaled])  # The pursuer's new velocity is along the slope of p, scaled by speed


    # Set action to pursuer change in x and y, scale action by acceleration (borrowed from distance_minimizing_fixed_strategy)
    agent.action.u = agent.accel * new_vel      
       
 
    # Plot to check that everything looks right
    if debug:
        plt.plot([x_p], [y_p], 'bo')  # pursuer
        plt.plot([x_p, x_c], [y_p, y_c], 'b--')
        plt.plot([x_p + new_vel[0]], [y_p + new_vel[1]], 'b*')
        plt.plot([x_e], [y_e], 'ro')  # evader
        plt.plot([x_e, x_c], [y_e, y_c], 'r')
        plt.plot([x_e + my_prey.state.p_vel[0]], [y_e + my_prey.state.p_vel[1]], 'm*')
        #plt.plot([x_p, x_e], [y_p, y_e], 'black')
        plt.plot([x_c], [y_c], 'gx')  # intersection
        plt.plot([x_p + agent.action.u[0]], [y_p + agent.action.u[1]], 'y*')  # Action?
        plt.title(vers + " bad_area: " + str(bad_area)+ " bad_order: " + str(bad_order) + " stat: " + str(abs(np.linalg.norm(B - A))<.00001) + " sf=" +str(sf) + '\nphi=' + str(math.floor(toDeg(phi))) + ' theta=' + str(math.floor(toDeg(theta))) + " \nE vel: "+str(my_prey.state.p_vel))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axhline(y=0, color='gray')
        plt.axvline(x=0, color='gray')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.grid()
        # plt.show()
        plt.savefig('C:/Users/erin.g.zaroukian/Desktop/Multi-Agent-Behaviors-Team/testPlots/' + str(counter) + "-" + str(counter%3) + '.png')  # Can't figure out how to access step
        plt.clf()

    if debug: counter += 1#For debugging plots
    return (agent.action)


def lion_and_man_pursuer(agent, world):
    """
    Basic recreation of the Lion and Man game where a predator lies in wait for a prey to come within a specified
    distance in which it then pursues

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    agent.action.u = np.zeros(world.dimension_position)

    global step
    global target_x
    global target_y

    # range at which the predator would pursue the prey
    distance_limit = 0.5

    # Collect prey agents
    prey = []
    for other in world.policy_agents:
        if not other.adversary:
            prey.append(other)

    # this is optimized for a single prey as of now
    single_prey = prey[0]

    # current prey position
    current_prey_position_x = single_prey.state.p_pos[0]
    current_prey_position_y = single_prey.state.p_pos[1]

    # current agent position
    current_agent_position_x = agent.state.p_pos[0]
    current_agent_position_y = agent.state.p_pos[1]

    # distance from prey to agent
    distance_prey_to_agent = math.sqrt(math.pow((current_prey_position_x - current_agent_position_x), 2) + math.pow(
        (current_prey_position_y - current_agent_position_y),2))

    # if prey is within desired range, attack (otherwise stay in place)
    if distance_prey_to_agent < distance_limit:
        target_x = current_prey_position_x - current_agent_position_x
        target_y = current_prey_position_y - current_agent_position_y
    else:
        target_x = (np.random.rand() - 0.5) / 1000
        target_y = (np.random.rand() - 0.5) / 1000

    x_n, y_n = target_x / np.linalg.norm(np.array([target_x, target_y])), target_y / np.linalg.norm(
        np.array([target_x, target_y]))

    # assigns actions
    agent.action.u[0] = x_n
    agent.action.u[1] = y_n

    # Scale action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


def utility_go_to(goal_x, goal_y, agent):
    """
    Utility method that moves the agent to a specified coordinate within a tolerance

    Args:
        goal_x (float): x position to move to
        goal_y (float): y position to move to
        agent (multiagent_particle_env.core.Agent): Agent object
    """
    global target_x
    global target_y

    global step
    tolerance = 0.1

    # current position
    current_agent_Position_x = agent.state.p_pos[0]
    current_agent_position_y = agent.state.p_pos[1]

    # range of possible positions for x,y
    range_high_x = goal_x + tolerance
    range_low_x = goal_x - tolerance
    range_high_y = goal_y + tolerance
    range_low_y = goal_y - tolerance

    # x,y to move to
    target_x = goal_x - current_agent_Position_x
    target_y = goal_y - current_agent_position_y

    # booleans of whether the agent is within acceptable range
    x_within_tolerance = range_low_x <= current_agent_Position_x <= range_high_x
    y_within_tolerance = range_low_y <= current_agent_position_y <= range_high_y

    if x_within_tolerance and y_within_tolerance:
        step += 1


def corners_fixed_strategy(agent, world):
    """
    Demo fixed prey policy in which it repeatedly moves around the corners of the field (acompanied by the utility
    go to method below)

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    agent.action.u = np.zeros(world.dimension_position)

    global step
    global target_x
    global target_y

    # top left
    if step == 0:
        utility_go_to(-1, 1, agent)

    # top right
    if step == 1:
        utility_go_to(1, 1, agent)

    # bottom right
    if step == 2:
        utility_go_to(1, -1, agent)

    # bottom left
    if step == 3:
        utility_go_to(-1, -1, agent)

    # top left (again)
    if step == 4:
        utility_go_to(-1, 1, agent)

    if step > 4:
        step = 0

    x_n, y_n = target_x / np.linalg.norm(np.array([target_x, target_y])), target_y / np.linalg.norm(
        np.array([target_x, target_y]))

    # assigns actions
    agent.action.u[0] = x_n
    agent.action.u[1] = y_n

    # Scale action by acceleration
    agent.action.u = agent.accel * agent.action.u

    return agent.action


###########################################
#           T-RECON Policies              #
###########################################

def fixed_turret(agent, world):
    """
    Fixed strategy for Turret Agent

    Args:
        agent (multiagent_particle_env.core.Agent): Agent object
        world (multiagent_particle_env.core.World): World object with agents and landmarks

    Returns:
        agent.action (multiagent_particle_env.core.Action): Agent action object
    """
    # Zero out agent action
    agent.action.u = np.zeros(world.dimension_position)

    # Collect known recon agents
    recon = []
    for other in world.policy_agents:
        if other.adversary and other.active and world.in_sense_region(agent, other):
            recon.append(other)

    if len(recon) == 0:
        # No recon agents seen, use random movement

        # Set current velocity to 0
        # agent.state.p_placeholder_vel = np.zeros(world.dimension_position)

        # Choose random direction
        direction = np.random.choice([1, -1])

        # Choose random accelaration and apply direction
        acceleration = np.random.uniform(0, agent.accel) * direction

        # Set action
        agent.action.u[0] = acceleration

    elif len(recon) == 1:
        # One recon agent seen, move in direction of shortest distance and max speed

        # Angle of recon agent
        recon_angle = pos_to_angle(recon[0].state.p_pos)

        # Determin direction
        direction, _ = calculate_shortest_polar_directon(agent.state.p_head_theta, recon_angle)

        # Choose random accelaration and apply direction
        acceleration = agent.accel * direction

        # Set action
        agent.action.u[0] = acceleration
    else:
        # More than One recon agent seen, determine closet recon agent,
        # then move in direction of shortest distance at max speed

        # Angle of recon agent
        recon_0_angle = pos_to_angle(recon[0].state.p_pos)
        recon_1_angle = pos_to_angle(recon[1].state.p_pos)

        # Determin direction for each recon agent
        recon_0_direction, recon_0_distance = calculate_shortest_polar_directon(agent.state.p_head_theta,
                                                                                recon_0_angle)

        recon_1_direction, recon_1_distance = calculate_shortest_polar_directon(agent.state.p_head_theta,
                                                                                recon_1_angle)

        # Determine which recon agent is closer and move in their direction
        if np.abs(recon_0_distance) == np.abs(recon_1_distance):
            direction = np.random.choice([1, -1])
        elif np.abs(recon_0_distance) < np.abs(recon_1_distance):
            direction = recon_0_distance
        else:
            direction = recon_1_distance

        # Choose random accelaration and apply direction
        acceleration = agent.accel * direction

        # Set action
        agent.action.u[0] = acceleration

    return agent.action
