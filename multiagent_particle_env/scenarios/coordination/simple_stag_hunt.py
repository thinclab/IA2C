# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_stag_hunt.py

Predator-prey

Communication: No
Competitive:   Yes

Good agents [Prey] (green) are faster and want to avoid being hit by adversaries [Predators] (red).
Adversaries are slower and want to hit good agents. Obstacles block the screen edges.

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import numpy as np

from colorama import Fore, Style
from random import shuffle
from scipy.spatial import distance

from multiagent_particle_env.core import World, Agent, Landmark
from multiagent_particle_env.scenario import BaseScenario
from multiagent_particle_env.utils import dict2namedtuple, normalize_adj

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


class Scenario(BaseScenario):
    """
    Define the world, reward, and observations for the scenario.
    """
    def __init__(self):
        """
        Initialize the scenario variables
        """
        self.args = None

        # Agent properties
        self.agent_obs = None
        self.agent_obs_size = None
        self.agent_move_block = None
        self.agent_share_space = None
        self.agent_share_qty = None
        self.agent_num_actions = None

        # World properties
        self.episode_limit = 200
        self.num_features = None
        self.num_predators = None
        self.num_prey = None
        self.num_obstacles = None
        self.observe_ids = None
        self.prey_capture_block = None
        self.prey_move_block = None
        self.prey_rest = 0.0
        self.state_shape = None
        self.state_size = None
        self.truncate_episodes = True
        self.world_shape = None
        self.world_shape_oversized = None

        # Reward properties
        self.modified_penalty_condition = None
        self.miscapture_punishment = None
        self.reward_stag = None
        self.reward_collision = None
        self.reward_time = None

        # Capture properties
        self.capture_freezes = False
        self.capture_terminal = False
        self.diagonal_capture = False
        self.remove_frozen = False

        # Episode properties
        self.ep_info = {}
        self.shared_reward = 0
        self.steps = 0
        self.terminal = False

        # Action properties
        self.action_labels = {'stay': 0, 'left': 1, 'right': 2, 'down': 3, 'up': 4, 'catch': 5}
        self.capture_action = False
        self.capture_action_conditions = 1
        self.capture_conditions = 0
        self.check_diagonal = False
        self.grid_actions = np.asarray([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]], dtype=np.int32)
        self.diagonal_actions = np.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int32)
        self.num_capture_actions = 1

        # Grid properties
        self.grid = None
        self.grid_oversized = None
        self.grid_positions = None
        self.world_positions = None
        self.num_positions = None
        self.num_entities = None

        # Extra observations
        self.observe_grid_pos = None
        self.observe_current_timestep = None

        # Grid render
        self._base_grid_img = None
        self.grid_viewer = None

        # Adjacency Matrix
        self.adj_mask_changed = False
        self.adj_mask = None
        self.fixed_adjacency_matrix = None
        self.use_adj_matrix = False
        self.use_adj_mask = False
        self.use_self_attention = False
        self.use_fixed_graph = False
        self.fixed_graph_topology = None
        self.proximal_distance = 2.0

    ###########################################
    #             World Functions             #
    ###########################################
    def make_world(self, args):
        """
        Construct the world for the scenario

        Returns:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        args = args
        if isinstance(args, dict):
            args = dict2namedtuple(args)

        self.args = args

        ###########################################
        #            Set properties               #
        ###########################################
        # Create world and set properties
        world = World()
        world.capture_action = args.capture_action
        world.num_capture_actions = args.num_capture_actions
        world.collaborative = True
        world.debug = args.debug
        world.discrete_action_input = True
        world.stag_hunt = True
        world.log_headers = ["Agent_Type", "Active", "gX", "gY", "X", "Y", "dX", "dY", "fX", "fY",
                             "Action", "Action_Label"]
        self.num_predators = args.num_predators
        self.num_prey = args.num_prey
        self.num_obstacles = args.num_obstacles

        # Capture properties
        self.capture_action = args.capture_action
        self.capture_action_conditions = args.capture_action_conditions
        self.capture_conditions = args.capture_conditions
        self.capture_freezes = args.capture_freezes
        self.capture_terminal = args.capture_terminal
        self.diagonal_capture = getattr(args, "diagonal_capture", False)
        self.remove_frozen = args.remove_frozen
        self.num_capture_actions = args.num_capture_actions

        # Update action_labels and grid_actions when num_capture_Actions is greater than 1
        if self.num_capture_actions > 1:
            self.action_labels = {'stay': 0, 'left': 1, 'right': 2, 'down': 3, 'up': 4}
            self.grid_actions = np.asarray([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]] +
                                           [[0, 0]] * self.num_capture_actions, dtype=np.int32)
            for i in range(self.num_capture_actions):
                self.action_labels[f'catch_{i}'] = i + 5

        # Reward properties
        self.modified_penalty_condition = getattr(args, "modified_penalty_condition", None)
        self.miscapture_punishment = args.miscapture_punishment
        self.reward_stag = args.reward_stag
        self.reward_collision = args.reward_collision
        self.reward_time = args.reward_time

        # World properties
        self.episode_limit = args.episode_limit
        self.observe_ids = getattr(args, "observe_ids", False)

        # Share space properties
        self.agent_share_space = getattr(args, "agent_share_space", False)
        self.agent_share_qty = getattr(args, "agent_share_qty", 1)

        remove_obstacles_feat = getattr(args, "remove_obstacles_feat", False)

        # Set number of features
        # 0=agents, 1=stag, 2=obstacle, [3=number of predators in space]
        if self.agent_share_space and self.observe_ids:
            # 0=agent_0, 1=agent_1, 2=agent_2, 3=stag, 4=obstacle, 5=number of predators in space
            # 0=agent_0, 1=agent_1, 2=stag, 3=obstacle, 4=number of predators in space
            self.num_features = self.agent_share_qty + 3
        elif self.agent_share_space:
            # 0=agents, 1=stag, 2=obstacle, [3=number of predators in space]
            self.num_features = 4
        elif remove_obstacles_feat:
            # 0=agents, 1=stag,
            self.num_features = 2
        else:
            # 0=agents, 1=stag, 2=obstacle
            self.num_features = 3

        self.prey_capture_block = np.asarray(getattr(args, "prey_capture_block", [0]), dtype=np.int16)
        self.prey_move_block = np.asarray(getattr(args, "prey_move_block", [0, 1, 2]), dtype=np.int16)
        self.prey_rest = args.prey_rest
        self.state_shape = [args.world_shape[0], args.world_shape[1], self.num_features]
        self.state_size = args.world_shape[0] * args.world_shape[1] * self.num_features
        self.truncate_episodes = getattr(args, "truncate_episodes", True)
        self.world_shape = np.array(args.world_shape, dtype=np.int16)
        self.world_shape_oversized = self.world_shape + (np.array(args.agent_obs, dtype=np.int16) * 2)

        # Agent properties
        self.agent_move_block = np.asarray(getattr(args, "agent_move_block", [0, 1, 2]), dtype=np.int16)

        if self.capture_action:
            self.agent_num_actions = 5 + self.num_capture_actions
        else:
            self.agent_num_actions = 5

        self.agent_obs = np.array(args.agent_obs, dtype=np.int16)
        self.agent_obs_size = ((2 * args.agent_obs[0]) + 1) * ((2 * args.agent_obs[1]) + 1) * self.num_features

        # Extra observations
        self.observe_grid_pos = getattr(args, "observe_grid_pos", False)
        if self.observe_grid_pos:
            self.agent_obs_size += 2

        self.observe_current_timestep = getattr(args, "observe_current_timestep", False)
        if self.observe_current_timestep:
            self.agent_obs_size += 1

        # Adjacency Matrix
        attention_type = getattr(args, "attention_type", None)
        adj_matrix = getattr(args, "use_adj_matrix", False)
        self.use_adj_matrix = True if attention_type is None and adj_matrix else False
        self.use_adj_mask = getattr(args, "use_adj_mask", False)
        self.fixed_graph_topology = getattr(args, "cg_topology", None)
        self.use_fixed_graph = False if self.fixed_graph_topology is None else True
        self.use_self_attention = getattr(args, "use_self_attention", True)
        self.proximal_distance = getattr(args, "proximal_distance", 2.0)

        if self.use_adj_matrix and self.use_adj_mask:
            self.adj_mask = np.ones((self.num_predators, self.num_predators))

        ###########################################
        #            Create entities              #
        ###########################################
        # Add predators
        for i in range(self.num_predators):
            world.agents.append(Agent())
            world.agents[i].index = i + 1
            world.agents[i].name = f'predator_agent_{i}'
            world.agents[i].adversary = True
            world.agents[i].accel = 20.0
            world.agents[i].collide = True
            world.agents[i].color = np.array([0.85, 0.35, 0.35, 1.0])
            world.agents[i].silent = True
            world.agents[i].size = 0.075
            world.agents[i].max_speed = None

        # Add prey
        for i in range(self.num_prey):
            world.random_agents.append(Agent())
            world.random_agents[i].index = i + 1
            world.random_agents[i].name = f'prey_agent_{i}'
            world.random_agents[i].adversary = False
            world.random_agents[i].accel = 20.0
            world.random_agents[i].collide = True
            world.random_agents[i].color = np.array([0.35, 0.85, 0.35, 1.0])
            world.random_agents[i].silent = True
            world.random_agents[i].size = 0.05
            world.random_agents[i].max_speed = None
            world.random_agents[i].movable = True

        # Add obstacles
        for i in range(self.num_obstacles):
            world.landmarks.append(Landmark())
            world.landmarks[i].index = i + 1
            world.landmarks[i].name = f'obstacle_{i}'
            world.landmarks[i].boundary = False
            world.landmarks[i].collide = True
            world.landmarks[i].color = np.array([0.25, 0.25, 0.25, 1.0])
            world.landmarks[i].movable = False
            world.landmarks[i].size = 0.2

        ###########################################
        #             Setup Grid Index            #
        ###########################################
        # Grid positions are indexed in y,x [row, column] order
        # World positions are returned in x,y order
        x = np.round(np.linspace(-.9, .9, self.world_shape[0]), decimals=1)
        y = np.round(np.linspace(.9, -.9, self.world_shape[1]), decimals=1)
        self.world_positions = np.flip(np.array(np.meshgrid(y, x)).T.reshape(self.world_shape[0],
                                                                             self.world_shape[1],
                                                                             world.dimension_position), axis=-1)

        # World positions are keyed in x,y order
        # Grid positions are returned in y,x [row, column] order
        x = np.array(range(self.world_shape[0]))
        y = np.array(range(self.world_shape[1]))
        grid_positions = np.array(np.meshgrid(x, y)).T.reshape(-1, world.dimension_position)
        self.grid_positions = dict(zip(tuple(map(tuple, self.world_positions.reshape(-1, world.dimension_position))),
                                       grid_positions.tolist()))

        self.num_positions = self.world_shape[0] * self.world_shape[1]
        self.num_entities = len(world.agents) + len(world.random_agents) + len(world.landmarks)

        ###########################################
        #          Set initial conditions         #
        ###########################################
        self.reset_world(world)

        return world

    def reset_world(self, world):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        ###########################################
        #        Initialize Shared Reward         #
        ###########################################
        self.ep_info = {}
        self.shared_reward = 0
        self.steps = 0
        self.terminal = False

        ###########################################
        #            Initialize Grid              #
        ###########################################
        self.grid = np.zeros((self.world_shape[0], self.world_shape[1], self.num_features), dtype=np.float32)
        self.grid_oversized = np.zeros((self.world_shape_oversized[0], self.world_shape_oversized[1],
                                        self.num_features), dtype=np.float32)

        ###########################################
        #   Set entity properties and positions   #
        ###########################################
        # Generate random indices for grid positions
        random_indices = np.random.choice(np.array(range(self.num_positions)), self.num_entities, replace=False)
        grid_keys = list(self.grid_positions.keys())

        # Set random initial states for agents
        for agent in world.agents:
            # Set agent properties
            agent.active = True
            agent.color[3] = 1.0

            # Choose grid key and move random indices forward
            grid_key = grid_keys[random_indices[0]]
            random_indices = random_indices[1:]

            agent.state.p_pos = np.array(grid_key)
            agent.state.g_pos = np.array(self.grid_positions[grid_key])
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

            # Place agent in grid
            if self.agent_share_space:
                if self.observe_ids:
                    if self.agent_share_qty == 2:
                        self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][0] = agent.index
                        self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][4] = 1
                        self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                            agent.state.g_pos[1] + self.agent_obs[1]][0] = agent.index
                        self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                            agent.state.g_pos[1] + self.agent_obs[1]][4] = 1
                    elif self.agent_share_qty == 3:
                        self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][0] = agent.index
                        self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][5] = 1
                        self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                            agent.state.g_pos[1] + self.agent_obs[1]][0] = agent.index
                        self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                            agent.state.g_pos[1] + self.agent_obs[1]][5] = 1
                    else:
                        raise ValueError(f"Observe ids is not supported for a "
                                         f"share space quantity of {self.agent_share_qty}")
                else:
                    self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][0] = 1
                    self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][3] = 1
                    self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                        agent.state.g_pos[1] + self.agent_obs[1]][0] = 1
                    self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                        agent.state.g_pos[1] + self.agent_obs[1]][3] = 1
            else:
                if self.observe_ids:
                    self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][0] = agent.index
                    self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                        agent.state.g_pos[1] + self.agent_obs[1]][0] = agent.index
                else:
                    self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][0] = 1
                    self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                        agent.state.g_pos[1] + self.agent_obs[1]][0] = 1

        if self.use_adj_matrix and self.use_adj_mask:
            self.adj_mask_changed = False
            self.adj_mask = np.ones((self.num_predators, self.num_predators))

        # Set random initial states for prey
        for agent in world.random_agents:
            # Set agent properties
            agent.active = True
            agent.color[3] = 1.0

            # Choose grid key and move random indices forward
            grid_key = grid_keys[random_indices[0]]
            random_indices = random_indices[1:]

            agent.state.p_pos = np.array(grid_key)
            agent.state.g_pos = np.array(self.grid_positions[grid_key])
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

            # Place prey in grid
            if self.agent_share_space and self.observe_ids:
                if self.agent_share_qty == 2:
                    self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][2] = 1
                    self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                        agent.state.g_pos[1] + self.agent_obs[0]][2] = 1

                elif self.agent_share_qty == 3:
                    self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][3] = 1
                    self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                        agent.state.g_pos[1] + self.agent_obs[0]][3] = 1

                else:
                    raise ValueError(f"Observe ids is not supported for a "
                                     f"share space quantity of {self.agent_share_qty}")
            else:
                self.grid[agent.state.g_pos[0], agent.state.g_pos[1]][1] = 1
                self.grid_oversized[agent.state.g_pos[0] + self.agent_obs[0],
                                    agent.state.g_pos[1] + self.agent_obs[0]][1] = 1

        # Set random initial states for landmarks and boundary landmarks
        for landmark in world.landmarks:
            # Set landmark properties
            # landmark.color = np.array([0.25, 0.25, 0.25, 1.0])

            # Choose grid key and move random indices forward
            grid_key = grid_keys[random_indices[0]]
            random_indices = random_indices[1:]

            landmark.state.p_pos = np.array(grid_key)
            landmark.state.g_pos = np.array(self.grid_positions[grid_key])
            landmark.state.p_vel = np.zeros(world.dimension_position)

            # Place landmark in grid
            if self.agent_share_space and self.observe_ids:
                if self.agent_share_qty == 2:
                    self.grid[landmark.state.g_pos[0], landmark.state.g_pos[1]][3] = 1
                    self.grid_oversized[landmark.state.g_pos[0] + self.agent_obs[0],
                                        landmark.state.g_pos[1] + self.agent_obs[0]][3] = 1

                elif self.agent_share_qty == 3:
                    self.grid[landmark.state.g_pos[0], landmark.state.g_pos[1]][4] = 1
                    self.grid_oversized[landmark.state.g_pos[0] + self.agent_obs[0],
                                        landmark.state.g_pos[1] + self.agent_obs[0]][4] = 1

                else:
                    raise ValueError(f"Observe ids is not supported for a "
                                     f"share space quantity of {self.agent_share_qty}")
            else:
                self.grid[landmark.state.g_pos[0], landmark.state.g_pos[1]][2] = 1
                self.grid_oversized[landmark.state.g_pos[0] + self.agent_obs[0],
                                    landmark.state.g_pos[1] + self.agent_obs[0]][2] = 1

    ###########################################
    #              Data Functions             #
    ###########################################
    def done(self, world):
        """
        Episode is done if all prey are captured, or if all predators are frozen,
        or if a single prey is captured when self.capture_terminal is True

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (bool) Terminal condition reached flag computed in apply_and_integrate_action()
        """
        return self.terminal

    def info(self, world):
        """
        Episode info

        Provides whether the episode reached it's timestep limit

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (dict) Episode info collected in apply_and_integrate_action()
        """
        return self.ep_info

    def logging(self, agent, world):
        """
        Collect data for logging.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (list) Data for logging
        """
        # Log elements
        agent_type = "predator" if agent.adversary else "stag"
        active = agent.active
        gx = agent.state.g_pos[0]
        gy = agent.state.g_pos[1]
        x = agent.state.p_pos[0]
        y = agent.state.p_pos[1]
        dx = agent.state.p_vel[0]
        dy = agent.state.p_vel[1]
        fx = agent.action.u[0]
        fy = agent.action.u[1]
        action = agent.action.discrete_raw
        grid_action = list(self.action_labels.keys())[action]

        log_data = [agent_type, active, gx, gy, x, y, dx, dy, fx, fy, action, grid_action]

        return log_data

    def observation(self, world):
        """
        Define the observations.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (np.array) Observations
            (np.array) Available actions
            (np.array) Grid State
        """
        obs_n = []
        available_actions_n = []
        for agent in world.agents:
            # Get available actions
            available_actions_n.append(self.get_avail_agent_actions(agent, world))

            obs = np.zeros((((2 * self.agent_obs[0]) + 1), ((2 * self.agent_obs[1]) + 1), self.num_features),
                           dtype=np.float32)

            if agent.active:
                obs = self.grid_oversized[agent.state.g_pos[0]:(agent.state.g_pos[0] + ((2 * self.agent_obs[0]) + 1)),
                                          agent.state.g_pos[1]:(agent.state.g_pos[1] + ((2 * self.agent_obs[1]) + 1)), :]
                obs = obs.reshape(-1)

                if self.observe_current_timestep:
                    timestep = self.steps/self.episode_limit
                    obs = np.concatenate([np.array([timestep], dtype=np.float32), obs])

                if self.observe_grid_pos:
                    # obs = np.concatenate([agent.state.g_pos, obs])
                    obs = np.concatenate([agent.state.g_pos/self.world_shape, obs])
            else:
                obs = obs.reshape(-1)

                # Add zeroes for extra observations if agent is not active
                if self.observe_current_timestep:
                    obs = np.concatenate([np.array([0.0], dtype=np.float32), obs])

                if self.observe_grid_pos:
                    obs = np.concatenate([np.array([0.0, 0.0], dtype=np.float32), obs])

            obs_n.append(obs)

        if self.use_adj_matrix:
            return np.array(obs_n, dtype=np.float32), self.get_agent_adj_matrix(world), np.array(available_actions_n), \
                self.grid.reshape(self.state_size)
        else:
            return np.array(obs_n, dtype=np.float32), np.array(available_actions_n), self.grid.reshape(self.state_size)

    def reward(self, world, shaped=False):
        """
        Reward is based on successfully capturing the prey in cooperation with a partner
        Else a penalty is incurred

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shaped (boolean): Specifies whether to use shaped reward, adds distance based increase and decrease.

        Returns:
            (float/int) Shared reward computed in apply_and_integrate_action()
        """
        return self.shared_reward

    ###########################################
    #                Utilities                #
    ###########################################
    def apply_and_integrate_action(self, world):
        """
        Apply action to agents and integrate physical state
        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        """
        self.shared_reward = 1.0 * self.reward_time
        for i in np.random.permutation(len(world.agents)):
            if world.agents[i].active:
                grid_position, collision = self.move_agent(world,
                                                           world.agents[i].state.g_pos,
                                                           world.agents[i].action.discrete_raw,
                                                           self.agent_move_block, move_agent_type=0,
                                                           agent_index=world.agents[i].index)

                world_position = self.world_positions[grid_position[0], grid_position[1]]

                if all(world_position == world.agents[i].state.p_pos):
                    # Position did not change
                    pass
                else:
                    # Update agent position state
                    world.agents[i].state.g_pos = grid_position
                    world.agents[i].state.p_pos = world_position

                if collision:
                    self.shared_reward = self.shared_reward + self.reward_collision

        for i in np.random.permutation(len(world.random_agents)):
            if world.random_agents[i].active:

                possible_actions = []
                neighbor_predators = 0
                movement_actions = np.array(range(4)) + 1
                diagonal_movement_actions = np.array(range(4))
                for action in movement_actions:
                    # Check for valid actions for the prey
                    if not self.move_agent(world, world.random_agents[i].state.g_pos, action, self.prey_move_block)[1]:
                        possible_actions.append(action)
                    # Check whether the prey is next to a predator
                    if self.move_agent(world, world.random_agents[i].state.g_pos, action,
                                       self.prey_capture_block)[1]:
                        neighbor_predators += 1

                if self.diagonal_capture:
                    for action in diagonal_movement_actions:
                        if self.check_prey_diagonal(world.random_agents[i].state.g_pos, action,
                                                    self.prey_capture_block):
                            neighbor_predators += 1

                if self.capture_action:
                    if self.num_capture_actions == 1:
                        num_catching_agents = 0
                        for agent in world.agents:
                            catching_agent = False
                            if agent.active and agent.action.capture[0]:
                                # If any movement action would end up on agent that agent can 'catch' prey
                                for action in movement_actions:
                                    position = world.random_agents[i].state.g_pos + self.grid_actions[action]
                                    if all(position == agent.state.g_pos):
                                        num_catching_agents += 1
                                        catching_agent = True
                                        break

                                if self.diagonal_capture and not catching_agent:
                                    for action in diagonal_movement_actions:
                                        position = world.random_agents[i].state.g_pos + self.diagonal_actions[action]
                                        if all(position == agent.state.g_pos):
                                            num_catching_agents += 1
                                            break

                        # If the number of neighboring agents that execute 'catch' >= condition, prey is captured
                        captured = num_catching_agents >= self.capture_action_conditions

                    else:
                        assert self.num_capture_actions == self.capture_action_conditions, \
                            f'The number of capture actions must be equal or less than the number ' \
                            f'of agents required to capture. {self.num_capture_actions} capture actions is ' \
                            f'not the same as the {self.capture_action_conditions} number of required capture agents.'

                        num_catching_agents = 0
                        capture_actions = np.zeros(self.num_capture_actions, dtype=bool)
                        for agent in world.agents:
                            catching_agent = False
                            if agent.active and np.any(agent.action.capture):
                                # If any movement action would end up on agent that agent can 'catch' prey
                                for action in movement_actions:
                                    position = world.random_agents[i].state.g_pos + self.grid_actions[action]
                                    if all(position == agent.state.g_pos):
                                        num_catching_agents += 1
                                        catching_agent = True
                                        capture_index = np.where(agent.action.capture == 1)[0][0]
                                        capture_actions[capture_index] = True
                                        break

                                if self.diagonal_capture and not catching_agent:
                                    for action in diagonal_movement_actions:
                                        position = world.random_agents[i].state.g_pos + self.diagonal_actions[action]
                                        if all(position == agent.state.g_pos):
                                            num_catching_agents += 1
                                            capture_index = np.where(agent.action.capture == 1)[0][0]
                                            capture_actions[capture_index] = True
                                            break

                        # If the number of neighboring agents that execute 'catch' >= condition, prey is captured
                        captured = (num_catching_agents >= self.capture_action_conditions) and np.all(capture_actions)

                    if self.modified_penalty_condition is None:
                        if num_catching_agents > 0 and not captured:
                            self.shared_reward += self.miscapture_punishment
                    else:
                        if num_catching_agents >= self.modified_penalty_condition and not captured:
                            self.shared_reward += self.miscapture_punishment

                else:
                    # Prey is caught when the number of predators is greater or equal to the capture conditions
                    captured = neighbor_predators >= self.capture_conditions

                    if self.modified_penalty_condition is None:
                        if neighbor_predators > 0 and not captured:
                            self.shared_reward += self.miscapture_punishment
                    else:
                        if neighbor_predators >= self.modified_penalty_condition and not captured:
                            self.shared_reward += self.miscapture_punishment

                if captured:
                    # Kill prey
                    world.random_agents[i].active = False

                    # Remove prey from grid
                    self.remove_agent(world, world.random_agents[i].state.g_pos, 1)

                    # Hide prey from world by setting to transparent
                    world.random_agents[i].color[3] = 0.1

                    # Terminate episode if any prey is captured
                    self.terminal = self.terminal or self.capture_terminal

                    # Add capture reward
                    self.shared_reward += self.reward_stag

                    if self.capture_freezes:
                        num_catching_agents = 0
                        capture_actions = np.zeros(self.num_capture_actions, dtype=bool)
                        for agent in world.agents:
                            if agent.active and (not self.capture_action or np.any(agent.action.capture)):
                                new_positions = agent.state.g_pos + self.grid_actions[:(self.agent_num_actions)]
                                overlaps = np.all(new_positions == world.random_agents[i].state.g_pos, axis=-1)

                                if self.diagonal_capture:
                                    new_diag_positions = agent.state.g_pos + self.diagonal_actions
                                    diag_overlaps = np.all(new_diag_positions == world.random_agents[i].state.g_pos,
                                                           axis=-1)
                                else:
                                    diag_overlaps = np.array([False])

                                if np.any(overlaps) or np.any(diag_overlaps):
                                    if self.num_capture_actions > 1:
                                        # Check if all capture actions are free and fill the one selected
                                        if np.all(~capture_actions):
                                            capture_index = np.where(agent.action.capture == 1)[0][0]
                                            capture_actions[capture_index] = True
                                        # End loop if all capture actions are filled
                                        # Should not occur but here for safety
                                        elif np.all(capture_actions):
                                            break
                                        # Skip to next agent if the same capture action is used
                                        elif np.where(agent.action.capture == 1)[0] in \
                                                np.where(capture_actions == 1)[0]:
                                            continue
                                        # Fill empty capture action
                                        else:
                                            capture_index = np.where(agent.action.capture == 1)[0][0]
                                            capture_actions[capture_index] = True

                                    # Freeze predator
                                    agent.active = False
                                    num_catching_agents += 1

                                    if self.remove_frozen:
                                        # Remove predator from grid
                                        self.remove_agent(world, agent.state.g_pos, 0, agent_index=agent.index)

                                        if self.use_adj_matrix and self.use_adj_mask:
                                            # Remove predator from adjacency matrix
                                            self.adj_mask[agent.index - 1, :] = 0
                                            self.adj_mask[:, agent.index - 1] = 0
                                            self.adj_mask_changed = True

                                        # Hide predator from world by setting to transparent
                                        agent.color[3] = 0.1

                                    if world.debug:
                                        print(f'{Fore.YELLOW} Freeze {agent.name} at grid position '
                                              f'{agent.state.g_pos}. Capture action: {agent.action.capture}. '
                                              f'Active agents: {world.active_agents} {Style.RESET_ALL}')

                                    # Don't freeze more agents than needed if extra agents
                                    # were part of the capture
                                    if num_catching_agents == self.capture_action_conditions:
                                        break

                    if world.debug:
                        print(f'{Fore.YELLOW} Captured {world.random_agents[i].name} at grid position '
                              f'{world.random_agents[i].state.g_pos}. Current reward {self.shared_reward}. '
                              f'Active agents: {world.active_agents}{Style.RESET_ALL}')

                else:
                    rest = (self.grid[world.random_agents[i].state.g_pos[0],
                                      world.random_agents[i].state.g_pos[1]][0] == 0) and \
                           (np.random.rand() < self.prey_rest) or (len(possible_actions) == 0)

                    if not rest:
                        action = possible_actions[np.random.randint(len(possible_actions))]

                        grid_position, _ = self.move_agent(world, world.random_agents[i].state.g_pos, action,
                                                           self.prey_move_block, move_agent_type=1)

                        world_position = self.world_positions[grid_position[0], grid_position[1]]

                        if all(world_position == world.random_agents[i].state.p_pos):
                            # Position did not change
                            pass
                        else:
                            # Update agent position state
                            world.random_agents[i].state.g_pos = grid_position
                            world.random_agents[i].state.p_pos = world_position

        prey_alive = len([agent.name for agent in world.random_agents if agent.active]) == 0
        predators_alive = len(world.active_agents) == 0

        # Terminate episode if all prey are caught or all agents are frozen
        self.terminal = self.terminal or prey_alive or predators_alive

        self.steps += 1

        if self.steps >= self.episode_limit:
            self.terminal = True
            self.ep_info["episode_limit"] = self.truncate_episodes
        else:
            self.ep_info["episode_limit"] = False

    def check_valid(self, grid_position):
        """
        Checks whether the position is valid in the environment
        Args:
            grid_position (np.array):

        Returns:
            (bool): Specifying whether the position is valid

        """
        return (0 <= grid_position[0] < self.world_shape[0]) and (0 <= grid_position[1] < self.world_shape[1])

    def environment_bound(self, grid_position):
        """
        Bound the position change by the environment bounds
        Args:
            grid_position (np.array):

        Returns:
            (np.array) Environment bounded position update

        """
        positions = np.minimum(grid_position, self.world_shape - 1)
        positions = np.maximum(positions, 0)
        return positions

    def get_avail_agent_actions(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:

        """
        if not agent.active:
            # All agents that are frozen can only perform the "stay" action
            available_actions = np.zeros(self.agent_num_actions, dtype=np.int32).tolist()
            available_actions[self.action_labels['stay']] = 1
        else:
            # Check position against all possible action changes
            new_positions = agent.state.g_pos + self.grid_actions[:self.agent_num_actions]

            # Only allow actions within the bounds of the environment
            allowed = np.logical_and(new_positions >= 0, new_positions < self.world_shape).all(axis=1)

            # Check that at least one action is available
            assert np.any(allowed), "No available action in the environment: this should never happen!"

            available_actions = [int(allowed[action]) for action in range(self.agent_num_actions)]

        # If the agent is not frozen, the 'catch' action is only available next to a prey
        if world.capture_action and agent.active:
            if self.num_capture_actions > 1:
                for i in range(self.num_capture_actions):
                    available_actions[self.action_labels[f'catch_{i}']] = 0
            else:
                available_actions[self.action_labels['catch']] = 0

            # Check with virtual move actions if there is a prey next to the agent
            possible_catch_actions = np.array(range(4)) + 1
            for action in possible_catch_actions:
                # Check for collisions with prey
                if self.agent_share_space and self.observe_ids:
                    if self.agent_share_qty == 2:
                        prey_collision_mask = np.asarray([1+1], dtype=np.int16)
                    elif self.agent_share_qty == 3:
                        prey_collision_mask = np.asarray([1+2], dtype=np.int16)
                    else:
                        raise ValueError(f"Observe ids is not supported for a "
                                         f"share space quantity of {self.agent_share_qty}")
                else:
                    prey_collision_mask = np.asarray([1], dtype=np.int16)

                if self.move_agent(world, agent.state.g_pos, action, prey_collision_mask)[1]:
                    if self.num_capture_actions > 1:
                        for i in range(self.num_capture_actions):
                            available_actions[self.action_labels[f'catch_{i}']] = 1
                    else:
                        available_actions[self.action_labels['catch']] = 1
                    break
        return available_actions

    def get_capture_agent_indices(self, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (np.array): Indices for predator agents in position for successful capture

        """
        movement_actions = np.array(range(4)) + 1
        diagonal_movement_actions = np.array(range(4))

        if self.num_capture_actions == 1:
            catching_agents = np.zeros(self.num_predators).astype(bool)
            for i in np.random.permutation(len(world.random_agents)):
                num_catching_agents = 0
                catching_agent_indicies = []
                if world.random_agents[i].active:
                    for j in np.random.permutation(len(world.agents)):
                        catching_agent = False
                        if world.agents[j].active and not catching_agents[j]:
                            # If any movement action would end up on agent that agent can 'catch' prey
                            for action in movement_actions:
                                position = world.random_agents[i].state.g_pos + self.grid_actions[action]
                                if all(position == world.agents[j].state.g_pos):
                                    num_catching_agents += 1
                                    catching_agent_indicies.append(j)
                                    catching_agent = True
                                    break

                            if self.diagonal_capture and not catching_agent:
                                for action in diagonal_movement_actions:
                                    position = world.random_agents[i].state.g_pos + self.diagonal_actions[action]
                                    if all(position == world.agents[j].state.g_pos):
                                        num_catching_agents += 1
                                        catching_agent_indicies.append(j)
                                        break

                        if num_catching_agents == self.capture_action_conditions:
                            break

                    if num_catching_agents == self.capture_action_conditions:
                        for index in catching_agent_indicies:
                            catching_agents[index] = True

            return np.where(catching_agents == True)[0]

        else:
            catching_agents = np.zeros(self.num_predators).astype(bool)
            catching_agents_action_index = np.zeros(self.num_predators).astype(int)
            for i in np.random.permutation(len(world.random_agents)):
                catching_agent_indicies = []
                capture_action_indices = list(range(self.num_capture_actions))
                shuffle(capture_action_indices)
                if world.random_agents[i].active:
                    for j in np.random.permutation(len(world.agents)):
                        catching_agent = False
                        if world.agents[j].active and not catching_agents[j]:
                            # If any movement action would end up on agent that agent can 'catch' prey
                            for action in movement_actions:
                                position = world.random_agents[i].state.g_pos + self.grid_actions[action]
                                if all(position == world.agents[j].state.g_pos):
                                    catching_agent = True
                                    capture_index = capture_action_indices.pop(0)
                                    catching_agent_indicies.append((j, capture_index))
                                    break

                            if self.diagonal_capture and not catching_agent:
                                for action in diagonal_movement_actions:
                                    position = world.random_agents[i].state.g_pos + self.diagonal_actions[action]
                                    if all(position == world.agents[j].state.g_pos):
                                        capture_index = capture_action_indices.pop(0)
                                        catching_agent_indicies.append((j, capture_index))
                                        break

                        if len(capture_action_indices) == 0:
                            break

                    if len(capture_action_indices) == 0:
                        for a_index, c_index in catching_agent_indicies:
                            catching_agents[a_index] = True
                            catching_agents_action_index[a_index] = c_index

            catching_agents = np.where(catching_agents == True)[0]
            return catching_agents, catching_agents_action_index[catching_agents]

    def check_prey_diagonal(self, grid_position, action, collision_mask):
        """
        Checks whether an entity is diagonally adjacent to a prey
        Can be used to test whether a prey is near for capture

        Args:
            grid_position (np.array):                   Grid position for the agent
            action (int):                               Action to execute
            collision_mask (np.array):                  Mask for entities that block movement

        Returns:

        """
        # compute hypothetical next position
        new_pos = grid_position + self.diagonal_actions[action]

        if self.check_valid(new_pos):
            # check for a collision with anything in the collision_mask
            found_at_new_pos = self.grid[new_pos[0], new_pos[1], :]
            collision = np.sum(found_at_new_pos[collision_mask]) > 0

            return collision

        else:
            return False

    def move_agent(self, world, grid_position, action, collision_mask, move_agent_type=None, agent_index=None):
        """
        Move entities on the grid
        Can be used to test whether a prey is near for capture

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            grid_position (np.array):                   Grid position for the agent
            action (int):                               Action to execute
            collision_mask (np.array):                  Mask for entities that block movement
            move_agent_type (None/int):                 Feature index for type of agent being moved
            agent_index (None/int):                     Agent index

        Returns:

        """
        # compute hypothetical next position
        new_pos = self.environment_bound(grid_position + self.grid_actions[action])

        # check for a collision with anything in the collision_mask
        found_at_new_pos = self.grid[new_pos[0], new_pos[1], :]
        collision = np.sum(found_at_new_pos[collision_mask]) > 0

        if collision:
            # if world.debug:
            #     print(f'{Fore.YELLOW} Collision at grid position {new_pos}. '
            #           f'Agent remaining at current grid position {grid_position}.{Style.RESET_ALL}')

            # No change in position if collision
            new_pos = grid_position

        elif move_agent_type is not None:
            if self.agent_share_space and self.observe_ids:
                if self.agent_share_qty == 2:
                    if move_agent_type == 0:
                        if (self.grid[new_pos[0], new_pos[1]][4] == 0) or (self.grid[new_pos[0], new_pos[1]][4] == 1):
                            self.grid[grid_position[0], grid_position[1]][4] -= 1
                            self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                                grid_position[1] + self.agent_obs[1]][4] -= 1

                            old_pos_index = np.where(self.grid[grid_position[0], grid_position[1]][:2] == agent_index)

                            self.grid[grid_position[0], grid_position[1]][old_pos_index] = 0
                            self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                                grid_position[1] + self.agent_obs[1]][old_pos_index] = 0

                            new_pos_index = int(self.grid[new_pos[0], new_pos[1]][4])

                            self.grid[new_pos[0], new_pos[1]][4] += 1
                            self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                                new_pos[1] + self.agent_obs[1]][4] += 1

                            self.grid[new_pos[0], new_pos[1]][new_pos_index] = agent_index
                            self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                                new_pos[1] + self.agent_obs[1]][new_pos_index] = agent_index

                            if self.grid[new_pos[0], new_pos[1]][4] == 2:
                                if world.debug:
                                    print(f'{Fore.YELLOW} {int(self.grid[new_pos[0], new_pos[1]][4])} '
                                          f'agents at grid position {new_pos}. {Style.RESET_ALL}')

                        else:
                            if world.debug:
                                print(f'{Fore.YELLOW} Too many agents at grid position {new_pos}. '
                                      f'Agent remaining at current grid position {grid_position}.{Style.RESET_ALL}')

                            # No change in position if space has too may agents
                            new_pos = grid_position

                    else:
                        # change the prey's or obstacle's state and position on the grid
                        self.grid[grid_position[0], grid_position[1]][move_agent_type + 1] = 0
                        self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                            grid_position[1] + self.agent_obs[1]][move_agent_type + 1] = 0

                        self.grid[new_pos[0], new_pos[1]][move_agent_type + 1] = 1
                        self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                            new_pos[1] + self.agent_obs[1]][move_agent_type + 1] = 1

                elif self.agent_share_qty == 3:
                    if move_agent_type == 0:
                        if (self.grid[new_pos[0], new_pos[1]][5] >= 0) and (self.grid[new_pos[0], new_pos[1]][5] <= 2):
                            self.grid[grid_position[0], grid_position[1]][5] -= 1
                            self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                                grid_position[1] + self.agent_obs[1]][5] -= 1

                            old_pos_index = np.where(self.grid[grid_position[0], grid_position[1]][:3] == agent_index)

                            self.grid[grid_position[0], grid_position[1]][old_pos_index] = 0
                            self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                                grid_position[1] + self.agent_obs[1]][old_pos_index] = 0

                            new_pos_index = int(self.grid[new_pos[0], new_pos[1]][5])

                            self.grid[new_pos[0], new_pos[1]][5] += 1
                            self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                                new_pos[1] + self.agent_obs[1]][5] += 1

                            self.grid[new_pos[0], new_pos[1]][new_pos_index] = agent_index
                            self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                                new_pos[1] + self.agent_obs[1]][new_pos_index] = agent_index

                            if self.grid[new_pos[0], new_pos[1]][5] > 1:
                                if world.debug:
                                    print(f'{Fore.YELLOW} {int(self.grid[new_pos[0], new_pos[1]][5])} '
                                          f'agents at grid position {new_pos}. {Style.RESET_ALL}')

                        else:
                            if world.debug:
                                print(f'{Fore.YELLOW} Too many agents at grid position {new_pos}. '
                                      f'Agent remaining at current grid position {grid_position}.{Style.RESET_ALL}')

                            # No change in position if space has too may agents
                            new_pos = grid_position
                    else:
                        # change the prey's or obstacle's state and position on the grid
                        self.grid[grid_position[0], grid_position[1]][move_agent_type + 2] = 0
                        self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                            grid_position[1] + self.agent_obs[1]][move_agent_type + 2] = 0

                        self.grid[new_pos[0], new_pos[1]][move_agent_type + 2] = 1
                        self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                            new_pos[1] + self.agent_obs[1]][move_agent_type + 2] = 1

                else:
                    raise ValueError(f"Observe ids is not supported for a "
                                     f"share space quantity of {self.agent_share_qty}")

            elif self.agent_share_space and (move_agent_type == 0):
                # change the agent's state and position on the grid

                if (self.grid[new_pos[0], new_pos[1]][3] == self.agent_share_qty):
                    if world.debug:
                        print(f'{Fore.YELLOW} Too many agents at grid position {new_pos}. '
                              f'Agent remaining at current grid position {grid_position}.{Style.RESET_ALL}')

                    # No change in position if space has too may agents
                    new_pos = grid_position

                else:
                    if self.grid[grid_position[0], grid_position[1]][3] > 1:
                        self.grid[grid_position[0], grid_position[1]][3] -= 1

                        self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                            grid_position[1] + self.agent_obs[1]][3] -= 1
                    else:
                        self.grid[grid_position[0], grid_position[1]][move_agent_type] = 0
                        self.grid[grid_position[0], grid_position[1]][3] = 0

                        self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                            grid_position[1] + self.agent_obs[1]][move_agent_type] = 0
                        self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                            grid_position[1] + self.agent_obs[1]][3] = 0

                    if self.grid[new_pos[0], new_pos[1]][3] >= 1:
                        self.grid[new_pos[0], new_pos[1]][3] += 1

                        self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                            new_pos[1] + self.agent_obs[1]][3] += 1
                    else:
                        self.grid[new_pos[0], new_pos[1]][move_agent_type] = 1
                        self.grid[new_pos[0], new_pos[1]][3] = 1

                        self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                            new_pos[1] + self.agent_obs[1]][move_agent_type] = 1
                        self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                            new_pos[1] + self.agent_obs[1]][3] = 1

                    if self.grid[new_pos[0], new_pos[1]][3] > 1:
                        if world.debug:
                            print(f'{Fore.YELLOW} {int(self.grid[new_pos[0], new_pos[1]][3])} '
                                  f'agents at grid position {new_pos}. {Style.RESET_ALL}')

            else:
                # change the agent's state and position on the grid
                self.grid[grid_position[0], grid_position[1]][move_agent_type] = 0
                self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                    grid_position[1] + self.agent_obs[1]][move_agent_type] = 0

                if self.observe_ids and (move_agent_type == 0):
                    self.grid[new_pos[0], new_pos[1]][move_agent_type] = agent_index
                    self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                        new_pos[1] + self.agent_obs[1]][move_agent_type] = agent_index
                else:
                    self.grid[new_pos[0], new_pos[1]][move_agent_type] = 1
                    self.grid_oversized[new_pos[0] + self.agent_obs[0],
                                        new_pos[1] + self.agent_obs[1]][move_agent_type] = 1

        return new_pos, collision

    def remove_agent(self, world, grid_position, move_agent_type, agent_index=None):
        """
        Remove entities on the grid
        Can be used to test whether a prey is near for capture

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            grid_position (np.array):                   Grid position for the agent
            move_agent_type (None/int):                 Feature index for type of agent being moved
            agent_index (None/int):                     Agent index
        """

        if self.agent_share_space and self.observe_ids:
            if self.agent_share_qty == 2:
                if move_agent_type == 0:
                    self.grid[grid_position[0], grid_position[1]][4] -= 1
                    self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                        grid_position[1] + self.agent_obs[1]][4] -= 1

                    old_pos_index = np.where(self.grid[grid_position[0], grid_position[1]][:2] == agent_index)

                    self.grid[grid_position[0], grid_position[1]][old_pos_index] = 0
                    self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                        grid_position[1] + self.agent_obs[1]][old_pos_index] = 0

                else:
                    # change the prey's or obstacle's state and position on the grid
                    self.grid[grid_position[0], grid_position[1]][move_agent_type + 1] = 0
                    self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                        grid_position[1] + self.agent_obs[1]][move_agent_type + 1] = 0

            elif self.agent_share_qty == 3:
                if move_agent_type == 0:
                    self.grid[grid_position[0], grid_position[1]][5] -= 1
                    self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                        grid_position[1] + self.agent_obs[1]][5] -= 1

                    old_pos_index = np.where(self.grid[grid_position[0], grid_position[1]][:3] == agent_index)

                    self.grid[grid_position[0], grid_position[1]][old_pos_index] = 0
                    self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                        grid_position[1] + self.agent_obs[1]][old_pos_index] = 0

                else:
                    # change the prey's or obstacle's state and position on the grid
                    self.grid[grid_position[0], grid_position[1]][move_agent_type + 2] = 0
                    self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                        grid_position[1] + self.agent_obs[1]][move_agent_type + 2] = 0

            else:
                raise ValueError(f"Observe ids is not supported for a "
                                 f"share space quantity of {self.agent_share_qty}")

        elif self.agent_share_space and (move_agent_type == 0):

                if self.grid[grid_position[0], grid_position[1]][3] > 1:
                    self.grid[grid_position[0], grid_position[1]][3] -= 1

                    self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                        grid_position[1] + self.agent_obs[1]][3] -= 1
                else:
                    self.grid[grid_position[0], grid_position[1]][move_agent_type] = 0
                    self.grid[grid_position[0], grid_position[1]][3] = 0

                    self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                        grid_position[1] + self.agent_obs[1]][move_agent_type] = 0
                    self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                        grid_position[1] + self.agent_obs[1]][3] = 0

        else:
            self.grid[grid_position[0], grid_position[1]][move_agent_type] = 0
            self.grid_oversized[grid_position[0] + self.agent_obs[0],
                                grid_position[1] + self.agent_obs[1]][move_agent_type] = 0

    def predator_agents(self, world):
        """
        Returns all agents that are adversaries in a list.

        Returns:
            (list) All the agents in the world that are adversaries.
        """
        return [agent for agent in world.agents if agent.adversary]

    def prey_agents(self, world):
        """
        Returns all agents that are not adversaries in a list.

        Returns:
            (list) All the agents in the world that are not adversaries.
        """
        return [agent for agent in world.random_agents if not agent.adversary]

    def print_grid(self, grid=None):
        """
        Print world grid

        Args:
            grid (None/np.array): Grid representing the world

        """
        if grid is None:
            grid = self.grid

        # Copy grid to prevent modifications from affecting original
        grid = grid.copy().astype('int64')

        if self.agent_share_space and self.observe_ids:
            if self.agent_share_qty == 2:
                for i in range(self.num_features):
                    if i == 0:
                        grid[:, :, i] *= 1000
                    elif i == 1:
                        grid[:, :, i] *= 10
                    elif i == 2:
                        grid[:, :, i] *= -1
                    elif i == 3:
                        grid[:, :, i] *= -2
                    else:
                        grid[:, :, i] *= 0
            elif self.agent_share_qty == 3:
                for i in range(self.num_features):
                    if i == 0:
                        grid[:, :, i] *= 100000
                    elif i == 1:
                        grid[:, :, i] *= 1000
                    elif i == 2:
                        grid[:, :, i] *= 10
                    elif i == 3:
                        grid[:, :, i] *= -1
                    elif i == 4:
                        grid[:, :, i] *= -2
                    else:
                        grid[:, :, i] *= 0
            else:
                raise ValueError(f"Observe ids is not supported for a "
                                 f"share space quantity of {self.agent_share_qty}")
        elif self.agent_share_space:
            # Add 1 to feature for each grid position occupied
            for i in range(self.num_features):
                if i == 3:
                    grid[:, :, i] *= 100
                else:
                    grid[:, :, i] *= i + 1
        else:
            if self.observe_ids:
                # Add 1 to feature for each grid position occupied
                for i in range(self.num_features):
                    if i == 0:
                        grid[:, :, i] *= i + -1
                    else:
                        grid[:, :, i] *= i + 1
            else:
                # Add 1 to feature for each grid position occupied
                for i in range(self.num_features):
                    grid[:, :, i] *= i + 1

        # Sum feature vector
        grid = np.sum(grid, axis=2)

        print(grid)

    def __check_neighbour_positions(self, world, grid_position):
        """

        Args:
            grid_position:

        Returns:

        """
        neighbour_positions = []
        for action in self.grid_actions[1:5]:
            pos = grid_position + action
            if self.check_valid(pos):
                neighbour_positions.append(pos)

        return neighbour_positions

    def grid_render(self, world, mode='human', save_filename=None, cell_size=35):
        """
        Render the grid environment

        Args:
            mode (string):

        Returns:
            results: ()
        """
        # if not self.stag_hunt_scenario:
        #     raise ValueError("Grid render is currently ony supported for the Stag Hunt scenario.")

        import copy
        from PIL import ImageColor

        from multiagent_particle_env.utils import draw_grid, fill_cell, draw_circle, write_cell_text

        # Grid Image Parameters
        agent_color = ImageColor.getcolor('blue', mode='RGB')
        agent_neighborhood_color = (186, 238, 247)
        prey_color = 'red'
        wall_color = 'black'

        # Get copy of base grid image
        if self._base_grid_img is None:
            self._base_grid_img = draw_grid(self.world_shape[0], self.world_shape[1], cell_size=cell_size, fill='white')
        img = copy.copy(self._base_grid_img)

        # Draw predators movement areas
        for agent in world.agents:
            if agent.active:
                for neighbour in self.__check_neighbour_positions(world, agent.state.g_pos):
                    fill_cell(img, neighbour, cell_size=cell_size, fill=agent_neighborhood_color, margin=0.1)
                fill_cell(img, agent.state.g_pos.tolist(), cell_size=cell_size, fill=agent_neighborhood_color,
                          margin=0.1)

        # Draw predators
        for agent in world.agents:
            if agent.active:
                draw_circle(img, agent.state.g_pos.tolist(), cell_size=cell_size, fill=agent_color)
                write_cell_text(img, text=str(agent.index), pos=agent.state.g_pos.tolist(), cell_size=cell_size,
                                fill='white', margin=0.4)

        # Draw prey
        for prey in world.random_agents:
            if prey.active:
                draw_circle(img, prey.state.g_pos.tolist(), cell_size=cell_size, fill=prey_color)
                write_cell_text(img, text=str(prey.index), pos=prey.state.g_pos.tolist(), cell_size=cell_size,
                                fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from multiagent_particle_env import rendering
            if self.grid_viewer is None:
                self.grid_viewer = rendering.SimpleImageViewer(maxwidth=700)
            self.grid_viewer.imshow(img, save_filename=save_filename)
            return self.grid_viewer.isopen

    def grid_render_with_attention(self, world, attention_weights=None, mode='human', save_filename=None,
                                   cell_size=35, line_size=20):
        """

        Args:
            world:
            attention_weights:
            mode:
            save_filename:

        Returns:

        """
        from PIL import Image, ImageDraw

        from multiagent_particle_env.utils import draw_attention_circle

        if attention_weights is not None:
            from multiagent_particle_env import rendering
            if self.grid_viewer is None:
                self.grid_viewer = rendering.SimpleImageViewer(maxwidth=1500)
            main_img = self.grid_render(world, mode='rgb_array', save_filename=save_filename, cell_size=cell_size)

            # Plot attention weights
            odd_imgs = []
            even_imgs = []
            for index, predator in enumerate(world.agents):
                img = Image.fromarray(main_img)
                start_col, start_row = predator.state.g_pos.tolist()
                start_x, start_y = (start_row + 0.5) * cell_size, (start_col + 0.5) * cell_size
                for i, agent in enumerate(world.agents):
                    if i == index:
                        draw_attention_circle(img, agent.state.g_pos.tolist(), cell_size=cell_size, fill=None,
                                              outline='green', radius=0.1,
                                              width=int(line_size * attention_weights[index][index]))
                    else:
                        if attention_weights[index][i] == 0:
                            fill = None
                        else:
                            fill = 'green'
                        end_col, end_row = agent.state.g_pos.tolist()
                        end_x, end_y = (end_row + 0.5) * cell_size, (end_col + 0.5) * cell_size
                        ImageDraw.Draw(img).line(((start_x, start_y), (end_x, end_y)),
                                                 fill=fill, width=int(line_size * attention_weights[index][i]))

                if (index % 2) == 0:
                    even_imgs.append(np.asarray(img))
                else:
                    odd_imgs.append(np.asarray(img))

            # Add placeholder image for odd number of agents
            if len(odd_imgs) != len(even_imgs):
                if len(odd_imgs) < len(even_imgs):
                    odd_imgs.append(np.asarray(self._base_grid_img))
                else:
                    even_imgs.append(np.asarray(self._base_grid_img))

            odd = np.concatenate(odd_imgs, axis=1)
            even = np.concatenate(even_imgs, axis=1)
            combined = np.append(even, odd, axis=0)
            self.grid_viewer.imshow(combined, save_filename=save_filename)

            return self.grid_viewer.isopen
        else:
            self.grid_render(world, mode='human', save_filename=save_filename)

    def get_agent_proximal_adj_matrix(self, world):
        """
        Compute adjacency matrix based on agent proximity

        Computes manhattan distance when p=1
        Diagonals are further away since movement is based on L,R,U,D
        scipy.spatial.distance.minkowski([0, 0], [1, 1], 1) = 2.0

        Computes chebyshev distance
        Assumes diagonal movement is possible
        scipy.spatial.distance.chebyshev([0,0], [1,1]) = 1

        Returns:
            adjacency_matrix (torch.Tensor):         Computed adjacency matrix
            softmax_adjacency_matrix (torch.Tensor): Computed adjacency matrix with softmax applied to the rows
        """
        adj_matrix = np.zeros((self.num_predators, self.num_predators))
        for i, i_agent in enumerate(world.agents):
            if i_agent.active:
                for j, j_agent in enumerate(world.agents):
                    if i == j:
                        if self.use_self_attention:
                            adj_matrix[i][j] = 1.0
                        else:
                            adj_matrix[i][j] = 0.0
                    else:
                        if j_agent.active and (distance.chebyshev(i_agent.state.g_pos, j_agent.state.g_pos) <=
                                               self.proximal_distance):
                            adj_matrix[i][j] = 1.0

        return np.expand_dims(adj_matrix, 0)

    def get_agent_fixed_adj_matrix(self):
        """

        Returns:

        """
        import multiagent_particle_env.cg_utils as cg_utils

        edges_list = cg_utils.cg_edge_list(self.fixed_graph_topology, self.num_predators)
        self.fixed_adjacency_matrix = cg_utils.set_cg_adj_matrix(edges_list, self.num_predators)

        if self.use_adj_mask:
            if self.use_self_attention:
                self.fixed_adjacency_matrix = (self.fixed_adjacency_matrix + np.eye(self.num_predators)) * self.adj_mask
            else:
                self.fixed_adjacency_matrix = self.fixed_adjacency_matrix * self.adj_mask
        else:
            if self.use_self_attention:
                self.fixed_adjacency_matrix = (self.fixed_adjacency_matrix + np.eye(self.num_predators))
            else:
                self.fixed_adjacency_matrix = self.fixed_adjacency_matrix

    def get_agent_adj_matrix(self, world):
        if not self.use_fixed_graph:
            return self.get_agent_proximal_adj_matrix(world)
        else:
            if self.fixed_adjacency_matrix is None:
                self.get_agent_fixed_adj_matrix()
            elif self.capture_freezes and self.adj_mask_changed and self.use_adj_mask:
                self.fixed_adjacency_matrix = self.fixed_adjacency_matrix * self.adj_mask
                self.adj_mask_changed = False
            else:
                pass

            return np.expand_dims(self.fixed_adjacency_matrix, 0)
