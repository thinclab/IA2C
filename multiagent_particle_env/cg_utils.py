# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cg_utils.py

Utilities for DICG
"""

import itertools
import numpy as np
import torch

from math import factorial
from random import randrange

from multiagent_particle_env.torch_utils import scatter_add

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2022'
__credits__ = ['Rolando Fernandez']
__license__ = ''
__version__ = '1.0.0'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


def cg_edge_list(cg_topology, num_agents):
    """
    Args:
        cg_topology (str, int, or list): Topology for the coordination graph
            if cg_topology is a string:
                Specifies edges for the following topologies:
                    empty - No connections'
                    trio  - Arrange agents in groups of 3
                    pairs - Arrange agents in groups of 2
                    line  - Arrange agents in a straight line
                    cycle - Arrange agents in a circle
                    star  - Arrange agents in a star around agent 0
                    full  - Connect all agents

            else if cg_topology is an integer:
                Specifies number of random edges to create for agent connections

            else if cg_topology is a list:
                Specifies the specific set of edges for agent connections

        num_agents (int):                Number of agents in the graph

    Returns:
        edges (np.array): List of agent connection edges for the coordination graph
    """
    edges = []

    if isinstance(cg_topology, str):
        if cg_topology == 'empty':
            pass
        elif cg_topology == 'trio':
            assert (num_agents % 3) == 0, "'trio' cg topology, is only for an odd number of agents divisible by 3 " \
                                          f"and will not work for the given '{num_agents}' number of agents"
            trios = list(zip(*[iter(range(num_agents))]*3))
            for trio in trios:
                edges.append([(i + trio[0], i + trio[0] + 1) for i in range(3 - 1)] + [(trio[-1], trio[0])])

            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        elif cg_topology == 'pairs':
            assert (num_agents % 2) == 0, "'pairs' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            edges = list(zip(*[iter(range(num_agents))]*2))
        elif cg_topology == 'double_line':
            assert (num_agents % 2) == 0, "'double_line' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            half = int(num_agents / 2)
            edges.append([(i, i + 1) for i in range(half - 1)])
            edges.append([(i, i + 1) for i in range(half, num_agents - 1)])
            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        elif cg_topology == 'line':
            edges = [(i, i + 1) for i in range(num_agents - 1)]
        elif cg_topology == 'double_cycle':
            assert (num_agents % 2) == 0, "'double_cycle' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            half = int(num_agents / 2)
            edges.append([(i, i + 1) for i in range(half - 1)] + [(half - 1, 0)])
            edges.append([(i, i + 1) for i in range(half, num_agents - 1)] + [(num_agents - 1, half)])
            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        elif cg_topology == 'cycle':
            edges = [(i, i + 1) for i in range(num_agents - 1)] + [(num_agents - 1, 0)]
        elif cg_topology == 'double_star':
            assert (num_agents % 2) == 0, "'double_star' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            half = int(num_agents / 2)
            edges.append([(0, i + 1) for i in range(half - 1)])
            edges.append([(half, i + 1) for i in range(half, num_agents - 1)])
            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        elif cg_topology == 'star':
            edges = [(0, i + 1) for i in range(num_agents - 1)]
        elif cg_topology == 'double_full':
            assert (num_agents % 2) == 0, "'double_full' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            half = int(num_agents / 2)
            edges.append([[(j, i + j + 1) for i in range(half - j - 1)] for j in range(half - 1)])
            edges.append([[(j, i + j + 1) for i in range(num_agents - j - 1)] for j in range(half, num_agents - 1)])
            # Flatten edges list
            edges = [edge for subgraph in edges for sublist in subgraph for edge in sublist]
        elif cg_topology == 'full':
            edges = [[(j, i + j + 1) for i in range(num_agents - j - 1)] for j in range(num_agents - 1)]
            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        else:
            raise ValueError("[dicg.cg_utils.cg_edge_list()]: "
                             "Parameter cg_topology must be one of the following when it is a string: "
                             "{'empty','trio','pairs','double_line','line','double_cycle','cycle',"
                             "'double_star','star','double_full','full'}")

    elif isinstance(cg_topology, int):
        if 0 <= cg_topology <= factorial(num_agents - 1):
            raise ValueError("[dicg.cg_utils.cg_edge_list()]: "
                             "Parameter cg_topology must be (<= n_agents!) when it is an integer")

        for i in range(cg_topology):
            edge_found = False
            while not edge_found:
                edge = (randrange(num_agents), randrange(num_agents))
                if (edge[0] != edge[1]) and (edge not in edges) and ((edge[1], edge[0]) not in edges):
                    edges.append(edge)
                    edge_found = True

    elif isinstance(cg_topology, list):
        # TODO: May need need to do check for duplicate edges
        if all([isinstance(edge, tuple) and (len(edge) == 2) and (all([isinstance(i, int) for i in edge]))
                for edge in cg_topology]):
            raise ValueError("[dicg.cg_utils.cg_edge_list()]: "
                             "Parameter cg_topology must be a list of int-tuples of length 2 with no duplicates "
                             "when it is a list specifying the agent connections.")
        edges = cg_topology

    else:
        raise ValueError("[dicg.cg_utils.cg_edge_list()]: "
                         f"{type(cg_topology)}, not supported for parameter cg_topology. "
                         "Parameter cg_topology must be either one of these string options "
                         "{'empty','trio','pairs','double_line','line','double_cycle','cycle',"
                         "'double_star','star','double_full','full'}, an integer for the number "
                         "of random edges that is (<= n_agents!), or a list of int-tuples for "
                         "each direct edge specification.")

    return np.array(edges)


def set_cg_edges(edges, num_agents):
    """
    Takes a list of tuples [0..n_agents)^2 and constructs the internal CG edge representation.

    Args:
        edges (list):     List of agent connection edges for the coordination graph
        num_agents (int): Number of agents in the graph

    Returns:
        edges_from (torch.Tensor): Tensor of from agents for agent connection edges in the coordination graph
        edges_to (torch.Tensor):   Tensor of to agents for agent connection edges in the coordination graph
        edges_n_in (torch.Tensor): Tensor of number of in edges per agents for agent connection edges
                                   in the coordination graph.
    """
    num_edges = len(edges)

    if num_edges == 0:
        edges_from = torch.zeros(num_edges, dtype=torch.long)
        edges_to = torch.zeros(num_edges, dtype=torch.long)
    else:
        edges_from = torch.from_numpy(edges[:, 0]).to(dtype=torch.long)
        edges_to = torch.from_numpy(edges[:, 1]).to(dtype=torch.long)

    # Count total number of edges for each agent
    edges_n_in = scatter_add(src=edges_to.new_ones(num_edges), index=edges_to, dim=0, dim_size=num_agents) + \
                 scatter_add(src=edges_to.new_ones(num_edges), index=edges_from, dim=0, dim_size=num_agents)

    return edges_from, edges_to, edges_n_in.float()


def set_cg_adj_matrix(edges, num_agents):
    """
    Takes a list of tuples [0..n_agents)^2 and constructs the internal CG edge representation as an adjacency matrix
    for use with Graph Neural Networks

    Args:
        edges (list):     List of agent connection edges for the coordination graph
        num_agents (int): Number of agents in the graph

    Returns:
        adjacency_matrix (torch.Tensor): Adjacency matrix representing the fixed coordination graph

    """
    adjacency_matrix = torch.zeros((num_agents, num_agents))
    for edge in edges:
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1

    return adjacency_matrix


def adj_matrix_to_hyperedge_index(adj_matrix, self_connection=False):
    """
    Takes the internal CG edge representation adjacency matrix for use with Graph Neural Networks and converts it
    into a hyperedge index representation for use with Hypergraph Neural Networks. Specifically, HypergraphConv
    in the torch geometric library.

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix representing the fixed coordination graph
        self_connection    (bool): Flag specifying whether to consider self connections (default: False)

    Returns:
        hyperedge_index (torch.Tensor): Hyperedge index tensor representing the hyperedge groups of the hypergraph

    """
    adj_matrix = torch.squeeze(adj_matrix)
    num_agents = len(adj_matrix)
    nodes = set(range(num_agents))
    node_sets = []
    for k in range(1, num_agents+1):
        for node_set in itertools.combinations(nodes, k):
            if len(node_set) == 1:
                if adj_matrix[node_set[0]][node_set[0]] == 1 and self_connection:
                    node_sets.append(node_set)
            else:
                if all(adj_matrix[i][j] == 1 for i in node_set for j in node_set if i != j):
                    node_sets.append(node_set)

    # Use list comprehension to create a new list with unique sublists
    unique_lists = [sublist for i, sublist in enumerate(node_sets) if sublist not in node_sets[:i]]

    # Remove subsets that exist in a larger set
    node_sets = [sublist for sublist in unique_lists if not any(set(sublist).issubset(other) for other in unique_lists  if other != sublist)]

    # Sum the total numbers of nodes in hyperedge groups
    num_hypernodes = sum([len(hyperedge_group) for hyperedge_group in node_sets])

    # Construct hyperedge index tensor
    hyperedge_index = torch.zeros((2, num_hypernodes), dtype=torch.int64)

    hypernode_counter = 0
    for edge_index, hyperedge in enumerate(node_sets):
        for node in hyperedge:
            hyperedge_index[0][hypernode_counter] = node
            hyperedge_index[1][hypernode_counter] = edge_index
            hypernode_counter += 1

    return hyperedge_index
