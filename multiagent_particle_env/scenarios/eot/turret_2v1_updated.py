# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
turret_2v1.py

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import numpy as np
from scipy.spatial import distance

from multiagent_particle_env.alternate_policies import fixed_turret
from multiagent_particle_env.core import World, Agent, TurretAgent
from multiagent_particle_env.scenario import BaseScenario
from multiagent_particle_env.utils import dict2namedtuple, pos_to_ang, pos_to_angle
from multiagent_particle_env.utils import torch_normalize_adj, tf_normalize_adj

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '1.0.0'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


class Scenario(BaseScenario):
    """
    Define the world, reward, and observations for the scenario.
    """

    def __init__(self):
        # Experiment arguments
        self.args = None

        # Debug verbose output
        self.debug = False

        # Optimal spawning distance
        self.optimal_dist = 0.1

        self.num_landmarks = 0
        self.num_adversaries = 2
        self.num_good_agents = 1
        self.num_agents = self.num_adversaries + self.num_good_agents

        # World properties
        self.agent_num_actions = 5
        self.episode_limit = None
        self.truncate_episodes = True
        self.num_features = None
        self.state_shape = None
        self.state_size = None
        self.agent_obs_size = 14

        # Action properties
        self.action_labels = {'stay': 0, 'left': 1, 'right': 2, 'down': 3, 'up': 4}

        # Episode properties
        self.ep_info = {}
        self.shared_reward = 0
        self.steps = 0
        self.terminal = False

        # Logging variables
        self.num_scores = None

        # Grid properties
        self.grid = None
        self.grid_positions = None
        self.world_positions = None
        self.num_positions = None
        self.num_entities = None

        # Grid render
        self._base_grid_img = None
        self.grid_viewer = None

        # Adjacency Matrix
        self.fixed_adjacency_matrix = None
        self.fixed_adjacency_matrix_softmax = None
        self.fixed_adjacency_matrix_normalized = None
        self.fixed_hyperedge_index = None
        self.use_adj_matrix = False
        self.use_hyperedge_index = False
        self.use_torch = False
        self.use_tf = False
        self.use_self_attention = False
        self.use_fixed_graph = False
        self.fixed_graph_topology = None
        self.proximal_distance = 2.0

    def make_world(self, args):
        """
        Construct the world

        Returns:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        args = args
        if isinstance(args, dict):
            args = dict2namedtuple(args)

        self.args = args

        # Display debug output if already slowing simulation to display visual
        self.debug = getattr(args, 'debug', False)

        ###########################################
        #            Set properties               #
        ###########################################
        # Create world and set properties
        world = World()
        world.discrete_action_input = True
        world.turret = True
        world.turret_cg = True
        world.dimension_communication = 2
        world.log_headers = ["Agent_Type", "Fixed", "Perturbed", "X", "Y", "dX", "dY", "fX", "fY", "entry_angle",
                             "theta", "theta_vel", "theta_x", "theta_y", "Score", "Collision"]

        # All agents and the HVT have the same base size
        # size is in mm?
        factor = 0.25
        reduction_factor = 0.5
        size = (0.025 * factor * 1.5) * reduction_factor

        # Speed scalar for adversaries
        # A value of 1.0 cause the speed to be the same as turret
        # attacker_v/turret_v
        nu = getattr(self.args, 'nu', .50)

        # Multiplied by all speeds (only purpose is to slow down agents between timesteps)
        speed_factor = 1.0 / 4.0

        # Multiplied by max speed to get acceleration
        accel_factor = 10

        # Attacker
        world.agents = [Agent() for _ in range(self.num_adversaries)]
        for i in range(self.num_adversaries):
            world.agents[i].name = 'attacker {}'.format(i)
            world.agents[i].adversary = True
            world.agents[i].collide = True
            world.agents[i].color = np.array([0.85, 0.35, 0.35])
            world.agents[i].silent = True
            world.agents[i].size = size
            world.agents[i].max_speed = nu * 10 * speed_factor
            world.agents[i].accel = world.agents[i].max_speed * accel_factor

        # Turret
        turret = TurretAgent()
        turret.name = 'turret'
        turret.color = np.array([0.25, 0.25, 0.25, .85])
        turret.has_sense = True
        turret.max_speed = 10 * speed_factor
        turret.accel = turret.max_speed * accel_factor
        turret.goal_radius = self.args.goal_radius * reduction_factor
        turret.sense_region = (self.args.goal_radius + self.args.visibility_from_goal) * reduction_factor
        turret.size = .1 * reduction_factor
        turret.silent = True
        turret.head_color = np.array([0.25, 0.25, 0.25])
        turret.head_size = size
        if args.fixed_turret:
            turret.action_callback = fixed_turret
            # turret.is_fixed_policy = True
        world.agents.append(turret)

        self.num_scores = {agent: 0 for agent in world.agents}

        # World properties
        self.agent_num_actions = 5
        self.episode_limit = args.episode_limit
        self.truncate_episodes = getattr(args, "truncate_episodes", True)
        self.num_features = 2
        self.state_shape = [args.world_shape[0], args.world_shape[1], self.num_features]
        self.state_size = args.world_shape[0] * args.world_shape[1] * self.num_features

        # Add boundary landmarks
        world.landmarks = world.landmarks + world.set_dense_boundaries()

        # Make initial conditions
        self.reset_world(world)

        # Adjacency Matrix
        attention_type = getattr(args, "attention_type", None)
        hypergraph = getattr(args, "hypergraph", False)
        self.use_adj_matrix = True if attention_type is None and not hypergraph else False
        self.use_hyperedge_index = True if attention_type is None and hypergraph else False
        self.fixed_graph_topology = getattr(args, "cg_topology", None)
        self.use_fixed_graph = False if self.fixed_graph_topology is None else True
        self.use_torch = getattr(args, "use_torch", False)
        self.use_tf = getattr(args, "use_tf", False)
        self.use_self_attention = getattr(args, "use_self_attention", False)
        self.proximal_distance = getattr(args, "proximal_distance", 2.0)

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
        self.turret_reward = 0
        self.steps = 0
        self.terminal = False

        ###########################################
        #            Initialize Grid              #
        ###########################################
        self.grid = np.zeros((self.state_shape[0], self.state_shape[1], self.num_features), dtype=np.float32)

        # Assumes grid size is odd, so that a center grid space can exist
        grid_center = np.array((self.state_shape[0], self.state_shape[1]), dtype=np.int32)
        grid_center = grid_center - 1
        grid_center = grid_center // 2
        grid_center = grid_center

        adversaries = self.adversaries(world)

        # Set initial conditions for turret
        # Turret stays in center of environment
        world.agents[-1].state.p_pos = np.zeros(world.dimension_position)
        world.agents[-1].state.p_vel = np.zeros(world.dimension_position)
        world.agents[-1].state.p_placeholder_vel = np.zeros(world.dimension_position)
        world.agents[-1].state.c = np.zeros(world.dimension_communication)

        # Place turret in center grid space
        self.grid[grid_center[0], grid_center[1]][1] = 1

        # Scenario 1: x,y to Theta with arctan logic
        # world.agents[-1].state.p_head_vel = np.zeros(world.dimension_position)
        # world.agents[-1].state.p_head_theta = np.zeros(1) + np.random.uniform(0, 2 * np.pi)
        # world.agents[-1].state.p_head_pos = np.concatenate([np.cos(world.agents[-1].state.p_head_theta) *
        #                                                     world.agents[-1].size,
        #                                                     np.sin(world.agents[-1].state.p_head_theta) *
        #                                                     world.agents[-1].size])
        # world.agents[-1].state.p_placeholder_pos = np.array([world.agents[-1].state.p_head_pos[0],
        #                                                      world.agents[-1].state.p_head_pos[1]])

        # Scenario 2: x to Theta with module 2PI logic
        world.agents[-1].state.p_head_vel = np.zeros(1)
        world.agents[-1].state.p_placeholder_pos = np.array([np.random.uniform(0, 2 * np.pi), 0])
        world.agents[-1].state.p_head_theta = np.array([world.agents[-1].state.p_placeholder_pos[0] % (2 * np.pi)])
        world.agents[-1].state.p_head_pos = np.concatenate([np.cos(world.agents[-1].state.p_head_theta) *
                                                            world.agents[-1].size,
                                                            np.sin(world.agents[-1].state.p_head_theta) *
                                                            world.agents[-1].size])

        # Set initial conditions for adversaries
        if self.args.spawn_optimal:
            # Optimally set adversaries spawn positions
            assert len(adversaries) == 2  # should only be two attackers (not sure where to put more or less)
            for i in range(2):
                # Sets attacker 0 self.opt_distance to left of sense region and attacker 1 opt_dist right
                # multiplies position by -1 if attacker 0 or positive 2 if 1
                adversaries[i].state.p_pos = \
                    np.array([(world.agents[-1].sense_region + self.optimal_dist) * (-1 + (2 * i)), 0])

                delta_pos = adversaries[i].state.p_pos - world.agents[-1].state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                inital_dist_to_goal = (dist - adversaries[i].size) - world.agents[-1].goal_radius

                # Not moving or communicating at start
                adversaries[i].state.p_vel = np.zeros(world.dimension_position)
                adversaries[i].state.c = np.zeros(world.dimension_communication)

                adversaries[i].inital_dist_to_goal = inital_dist_to_goal
                adversaries[i].prev_dist_to_goal = None
                adversaries[i].new_dist_to_goal = None
                adversaries[i].seen_by_turret = False
                adversaries[i].prev_dist = 2 * np.pi
                adversaries[i].new_dist = 2 * np.pi
                adversaries[i].hit = False
                adversaries[i].touched_turret = False
                adversaries[i].active = True
                adversaries[i].color = np.array([0.85, 0.35, 0.35])
        else:
            for adversary in adversaries:
                # Set random initial states for agents
                adversary.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
                if self.args.spawn_outside_sense:
                    while world.in_sense_region(world.agents[-1], adversary):
                        adversary.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)

                delta_pos = adversary.state.p_pos - world.agents[-1].state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                inital_dist_to_goal = (dist - adversary.size) - world.agents[-1].goal_radius

                # Not moving or communicating at start
                adversary.state.p_vel = np.zeros(world.dimension_position)
                adversary.state.c = np.zeros(world.dimension_communication)

                adversary.inital_dist_to_goal = inital_dist_to_goal
                adversary.prev_dist_to_goal = None
                adversary.new_dist_to_goal = None
                adversary.seen_by_turret = False
                adversary.prev_dist = 2 * np.pi
                adversary.new_dist = 2 * np.pi
                adversary.hit = False
                adversary.touched_turret = False
                adversary.active = True
                adversary.color = np.array([0.85, 0.35, 0.35])

    def calculate_win_lose_conditions(self, world):
        """
        Calculate the win lose conditions for the scenario
        and set hit and touched_turret parameters for adversaries.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        """
        turrets = self.good_agents(world)
        adversaries = self.adversaries(world)

        self.steps += 1

        for agent in turrets:
            for adv in adversaries:
                if not adv.active:
                    # attacker cannot be hit so no need to perform calculations
                    continue

                # Update angles for turret reward function

                # record previous distance between agent and turret
                adv.prev_dist = adv.new_dist

                # get attacker angle
                attacker_angle = pos_to_angle(adv.state.p_pos)

                # get distance
                adv.new_dist = attacker_angle - agent.state.p_head_theta
                if adv.new_dist < 0:
                    adv.new_dist += 2 * np.pi

                # new_dist is angular distance from turret to attacker ccw
                # if distance is larger than pi convert to cw and negate angle to reflect direction
                if adv.new_dist > np.pi:
                    adv.new_dist -= 2 * np.pi

                assert -np.pi < adv.new_dist <= np.pi, "Angular Distance greater than pi, " \
                                                       "Adversary New Dist: {} " \
                                                       "Adversary Angle: {} " \
                                                       "Turret Angle: {}".format(adv.new_dist,
                                                                                 attacker_angle,
                                                                                 agent.state.p_head_theta)

                # Adversary distance from origin
                dist_origin = np.sqrt(np.sum(np.square(adv.state.p_pos)))

                # If attacker has been see and there is a sign change in the distance between the turret and attacker
                # then there was a cross paths
                if adv.seen_by_turret and (abs(adv.prev_dist) <= np.pi) and \
                        ((adv.new_dist == 0) or (((adv.prev_dist < 0) != (adv.new_dist < 0)) and
                                                 (abs(adv.new_dist) < (np.pi / 2)))):
                    adv.hit = True

                elif world.is_collision(adv, agent) or (dist_origin <= agent.goal_radius):
                    adv.touched_turret = True

    def reward(self, world, dense=False, shared=True):
        """
        Reward is based on adversary agents not being hit by the turret.

        Turret is negatively rewarded if adversaries enter goal region.

        Adversaries are rewarded entering turret goal region.


        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            dense (boolean): Specifies whether to use dense reward
            shared (boolean): Specifies whether to use shared reward

        Returns:
            If agent is adversary:
                self.adversary_reward() result
            Else:
                self.agent_reward() result
        """
        # Update win lose conditions
        # Only do this once per timestep
        self.calculate_win_lose_conditions(world)

        self.shared_reward = self.adversary_reward(world, shared)
        self.turret_reward += self.agent_reward(world)

        if self.args.fixed_turret:
            return self.shared_reward
        else:
            return [self.shared_reward, self.turret_reward]

    def agent_reward(self, world):
        """
        Turret reward

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (float) Total agent reward
        """
        reward = 0
        adversaries = self.adversaries(world)

        for adv in adversaries:
            if not adv.active:
                # attacker cannot be hit so no need to perform calculations
                continue

            # Calculate reward for turret
            # If the adversary has both hit and touched the turret
            # the tie goes to the turret
            if adv.hit:
                reward += self.args.reward_turret_hit
                # agent.score += 1
                adv.active = False
                adv.color = np.array([.545, .365, .176])
            elif adv.touched_turret:
                reward -= self.args.reward_turret_touch
                adv.active = False
                adv.color = np.array([0.35, 0.85, 0.35])

        return reward

    def adversary_reward(self, world, shared):
        """
        Recon reward

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shared (boolean): Specifies whether to use shared reward

        Returns:
            (float) Total agent reward
        """
        reward = 0
        adversaries = self.adversaries(world)

        if shared:
            # Compute Shared Reward
            for adv in adversaries:
                if not adv.active:
                    # Attacker cannot achieve anymore reward
                    continue

                # Determine if agent left the screen and assign penalties
                for coordinate_position in range(world.dimension_position):
                    reward -= world.bound(abs(adv.state.p_pos[coordinate_position]))

                # Calculate reward for adversary
                # If the adversary has both hit and touched the turret
                # the tie goes to the turret
                if adv.hit:
                    reward -= self.args.reward_agent_hit
                    adv.score -= 1
                elif adv.touched_turret:
                    reward += self.args.reward_agent_touch
                    adv.score += 1

        return reward

    def observation(self, world):
        """
        Define the observations.
        all agents can only see active attackers

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (np.array) Observations array with the velocity of the agent, the position of the agent,
                       distance to all landmarks in the agent's reference frame,
                       distance to all other agents in the agent's reference frame,
                       and the velocities of the good agents.
        """
        agents = world.policy_agents
        turret_entities = world.scripted_agents
        adversaries = self.adversaries(world)
        turret_obs = []
        agent_obs = []

        for turret in turret_entities:
            other_pos = []
            other_vel = []
            # Turret
            # [angular velocity, 0.0, turret head x, turret head y, turret body position x, turret body position y,
            # adv 0 x, adv 0 y, adv 1 x, adv 1 y, adv 0 vx, adv 0 vy, adv 1 vx, adv 1 vy]

            # allow for defender to see in turret sense region
            # if attacker goes into turret sense region then defender can see it for the rest of the episode
            for adversary in adversaries:
                if adversary.active and (not adversary.seen_by_turret) and world.in_sense_region(turret, adversary):
                    # First entry into turret sense region
                    adversary.seen_by_turret = True
                    adversary.initial_entry = True

                if adversary.active and adversary.seen_by_turret:
                    # Attacker position relative to HVT
                    other_pos.append(adversary.state.p_pos - turret.state.p_pos)

                    # Attacker velocity
                    other_vel.append(adversary.state.p_vel)

                else:
                    # Pad if attacker not seen or not active
                    other_pos.append(np.zeros(2))
                    other_vel.append(np.zeros(2))

            if not self.args.fixed_turret:
                # 2 + 2 + 2 + 4 + 4 = 14
                turret_obs.append(np.concatenate([np.array([turret.state.p_head_vel[0], 0.0])] +
                                                 [turret.state.p_head_pos] + [turret.state.p_pos] +
                                                 other_pos + other_vel).astype(np.float32))

        for agent in agents:
            if not agent.adversary:
                continue

            other_pos = []
            other_vel = []
            # Recon agents
            # [ vel x, vel y, pos x, pos y,
            # turret body x, turret body y, turret head x, turret head y, partner x, partner y,
            # turret head vel, 0.0, partner vx, partner vy ]

            turret_pos = []
            turret_head_pos = []
            turret_head_vel = []
            turrets = [agent for agent in world.agents if isinstance(agent, TurretAgent)]
            for turret in turrets:
                turret_pos.append(turret.state.p_pos - agent.state.p_pos)
                turret_head_pos.append(turret.state.p_head_pos - agent.state.p_pos)
                turret_head_vel.append(np.array([turret.state.p_head_vel, 0.0]))

            for other in adversaries:
                if other is agent:
                    # Don't add current agent to other agents
                    continue
                elif other.active:
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    other_vel.append(other.state.p_vel)
                else:
                    # can't see non active agents
                    other_pos.append(np.zeros(2))
                    other_vel.append(np.zeros(2))

            # 2 + 2 + 2 + 2 + 2 + 2 + 2 = 14
            agent_obs.append((np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + turret_pos + turret_head_pos +
                                             other_pos + turret_head_vel + other_vel).astype(np.float32)))

        if self.args.fixed_turret:
            agent_obs.extend(turret_obs)

        return agent_obs, self.get_agent_adj_matrix(world), \
            self.get_avail_agent_actions(world), self.grid.reshape(self.state_size)

    def done(self, world):
        """
        not all touch Terminal conditions:
            1) all attackers are hit
            2) an attacker touches the turret/HVT

        all touch terminal condition:
            ever attacker has: a) been hit b) touched HVT

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (bool) Terminal condition reached flag
        """
        adversaries = self.adversaries(world)

        if self.steps == (self.episode_limit - 1):
            self.ep_info["episode_limit"] = self.truncate_episodes
            return True
        elif self.args.all_touch:
            # All attackers must have been hit or have touched the turret
            self.ep_info["episode_limit"] = False
            return all([not adversary.active for adversary in adversaries])
        else:
            # One attacker must have touched the turret or all attackers must have been hit
            self.ep_info["episode_limit"] = False
            return any([adversary.touched_turret and not adversary.hit for adversary in adversaries]) or \
                   all([adversary.hit for adversary in adversaries])

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
        agent_type = ""
        fixed = agent.is_fixed_policy
        perturbed = agent.is_perturbed_policy
        x = agent.state.p_pos[0]
        y = agent.state.p_pos[1]
        dx = agent.state.p_vel[0]
        dy = agent.state.p_vel[1]
        fx = agent.action.u[0]
        fy = agent.action.u[1]
        collision = 0
        entry_angle = None
        theta = None
        theta_vel = None
        theta_x = None
        theta_y = None

        # Check for collisions
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if agent in good_agents:
            agent_type = "Turret"
            score = agent.score
            agent.score = 0
            theta = agent.state.p_head_theta
            theta_vel = agent.state.p_head_vel
            theta_x = agent.state.p_head_pos[0]
            theta_y = agent.state.p_head_pos[1]
            for adv in adversaries:
                if world.is_collision(agent, adv):
                    collision += 1
        elif agent in adversaries:
            agent_type = "Recon"
            score = agent.score
            agent.score = 0
            if agent.initial_entry:
                agent.initial_entry = False
                entry_angle = pos_to_angle(agent.state.p_pos)
            for ga in good_agents:
                if world.is_collision(agent, ga):
                    collision += 1
        else:
            collision = None

        log_data = [agent_type, fixed, perturbed, x, y, dx, dy, fx, fy,
                    entry_angle, theta, theta_vel, theta_x, theta_y, score, collision]

        return log_data

    def good_agents(self, world):
        """
        Returns all agents that are not adversaries in a list.

        Returns:
            (list) All the agents in the world that are not adversaries.
        """
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        """
        Returns all agents that are adversaries in a list.

        Returns:            (list) All the agents in the world that are adversaries.
        """
        return [agent for agent in world.agents if agent.adversary]

    def get_avail_agent_actions(self, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:

        """
        agents = world.policy_agents
        available_actions_n = []
        for agent in agents:
            if not agent.active:
                # All agents that are frozen can only perform the "stay" action
                available_actions = np.zeros(self.agent_num_actions, dtype=np.int32).tolist()
                available_actions[self.action_labels['stay']] = 1
            else:
                available_actions = np.ones(self.agent_num_actions, dtype=np.int32).tolist()

            available_actions_n.append(available_actions)

        return np.array(available_actions_n)

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
        if self.use_torch:
            import torch
            from multiagent_particle_env.utils import sparse_mx_to_torch_sparse_tensor

            adj_matrix = torch.zeros((self.num_predators, self.num_predators)).to(dtype=torch.float32)
            for i, i_agent in enumerate(world.agents):
                if i_agent.active:
                    for j, j_agent in enumerate(world.agents):
                        if i == j:
                            if self.use_self_attention:
                                adj_matrix[i][j] = 1.0
                            else:
                                adj_matrix[i][j] = 0.0
                        else:
                            if j_agent.active and (distance.chebyshev(i_agent.state.g_pos, j_agent.state.g_pos) <= self.proximal_distance):
                                adj_matrix[i][j] = 1.0

            if self.use_self_attention:
                # Laplacian modified norm
                normalized_adj_matrix = torch_normalize_adj(adj_matrix)
            else:
                # Laplacian modified norm
                normalized_adj_matrix = torch_normalize_adj(adj_matrix + torch.eye(self.num_predators))

            normalized_adj_matrix = torch.unsqueeze(normalized_adj_matrix, 0)
            softmax_func = torch.nn.Softmax(dim=-1)
            adj_matrix = torch.unsqueeze(adj_matrix, 0)
            return adj_matrix, softmax_func(adj_matrix), normalized_adj_matrix

        elif self.use_tf:
            import tf
            from multiagent_particle_env.utils import sparse_mx_to_tf_sparse_tensor

            adj_matrix = tf.zeros((self.num_predators, self.num_predators), dtype=tf.float32)
            for i, i_agent in enumerate(world.agents):
                if i_agent.active:
                    for j, j_agent in enumerate(world.agents):
                        if i == j:
                            if self.use_self_attention:
                                adj_matrix[i][j] = 1.0
                            else:
                                adj_matrix[i][j] = 0.0
                        else:
                            if j_agent.active and (distance.chebyshev(i_agent.state.g_pos, j_agent.state.g_pos) <= self.proximal_distance):
                                adj_matrix[i][j] = 1.0

            if self.use_self_attention:
                # Laplacian modified norm
                normalized_adj_matrix = tf_normalize_adj(adj_matrix)
            else:
                # Laplacian modified norm
                normalized_adj_matrix = tf_normalize_adj(adj_matrix + tf.eye(self.num_predators, self.num_predators))

            normalized_adj_matrix = tf.expand_dims(normalized_adj_matrix, 0)
            adj_matrix = tf.expand_dims(adj_matrix, 0)
            softmax_func = tf.nn.softmax(adj_matrix, axis=-1)
            return adj_matrix, softmax_func, normalized_adj_matrix

        else:
            raise ValueError("Creation of proximal adjacency matrix is only supported for "
                             "either tensorflow or torch. Neither is currently selected for use.")

    def get_agent_fixed_adj_matrix(self):
        """

        Returns:

        """
        import dicg.cg_utils as cg_utils

        edges_list = cg_utils.cg_edge_list(self.fixed_graph_topology, self.num_adversaries)
        self.fixed_adjacency_matrix = cg_utils.set_cg_adj_matrix(edges_list, self.num_adversaries)

        if self.use_torch:
            import torch
            from multiagent_particle_env.utils import sparse_mx_to_torch_sparse_tensor

            if self.use_self_attention:
                self.fixed_adjacency_matrix = self.fixed_adjacency_matrix + torch.eye(self.num_adversaries)
                self.fixed_adjacency_matrix_normalized = torch_normalize_adj(self.fixed_adjacency_matrix)
            else:
                self.fixed_adjacency_matrix_normalized = torch_normalize_adj(self.fixed_adjacency_matrix +
                                                                             torch.eye(self.num_adversaries))

            self.fixed_adjacency_matrix = torch.unsqueeze(self.fixed_adjacency_matrix, 0)
            self.fixed_adjacency_matrix = self.fixed_adjacency_matrix.to(dtype=torch.float32)

            softmax_func = torch.nn.Softmax(dim=-1)
            self.fixed_adjacency_matrix_softmax = softmax_func(self.fixed_adjacency_matrix)

            self.fixed_adjacency_matrix_normalized = torch.unsqueeze(self.fixed_adjacency_matrix_normalized, 0)

        elif self.use_tf:
            import tensorflow as tf
            from multiagent_particle_env.utils import sparse_mx_to_tf_sparse_tensor

            if self.use_self_attention:
                self.fixed_adjacency_matrix = self.fixed_adjacency_matrix + tf.eye(self.num_adversaries,
                                                                                   self.num_adversaries)
                self.fixed_adjacency_matrix_normalized = tf_normalize_adj(self.fixed_adjacency_matrix)
            else:
                self.fixed_adjacency_matrix_normalized = tf_normalize_adj(self.fixed_adjacency_matrix +
                                                                          tf.eye(self.num_adversaries,
                                                                                 self.num_adversaries))

            self.fixed_adjacency_matrix = tf.expand_dims(self.fixed_adjacency_matrix, 0)
            self.fixed_adjacency_matrix = tf.cast(self.fixed_adjacency_matrix, tf.float32)

            softmax_func = tf.nn.softmax(self.fixed_adjacency_matrix, axis=-1)
            self.fixed_adjacency_matrix_softmax = softmax_func

            self.fixed_adjacency_matrix_normalized = tf.expand_dims(self.fixed_adjacency_matrix_normalized, 0)

        else:
            raise ValueError("Creation of fixed adjacency matrix is only supported for "
                             "either tensorflow or torch. Neither is currently selected for use.")

    def get_agent_adj_matrix(self, world):
        if not self.use_fixed_graph:
            return self.get_agent_proximal_adj_matrix(world)
        else:
            if self.fixed_adjacency_matrix is None:
                self.get_agent_fixed_adj_matrix()

            return self.fixed_adjacency_matrix, self.fixed_adjacency_matrix_softmax, self.fixed_adjacency_matrix_normalized

    def get_agent_hyperedge_index(self, world):
        if self.use_torch:
            import torch

            from dicg.cg_utils import adj_matrix_to_hyperedge_index

            if not self.use_fixed_graph:
                adj_matrix = self.get_agent_proximal_adj_matrix(world)[0]
                return torch.unsqueeze(adj_matrix_to_hyperedge_index(adj_matrix), 0), adj_matrix

            else:
                if self.fixed_adjacency_matrix is None:
                    self.get_agent_fixed_adj_matrix()
                    self.fixed_hyperedge_index = torch.unsqueeze(
                        adj_matrix_to_hyperedge_index(self.fixed_adjacency_matrix), 0)
                return self.fixed_hyperedge_index, self.fixed_adjacency_matrix

        else:
            raise ValueError("Creation of hyperedge index is only supported for torch."
                             "Which, is not currently selected for use.")
