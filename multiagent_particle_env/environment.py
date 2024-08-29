# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
environment.py

Environment for all agents in the multi-agent particle world.

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import gym
import numpy as np

from multiagent_particle_env.core import TurretAgent

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


class MultiAgentEnv(gym.Env):
    """
    Environment for all agents in the multi-agent particle world.

    Currently, code assumes that no agents will be created/destroyed at runtime!
    """
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, arglist, logger, reset_callback=None, reward_callback=None, observation_callback=None,
                 logging_callback=None, info_callback=None, done_callback=None, shared_viewer=True):
        """
        Args:
            world (multiagent_particle_env.core.World): World object containing all the entities of a specific scenario
            arglist (argparse object):  Parsed commandline arguments
            logger (multiagent_particle_env.logger.Logger): Logger object for logging data from the world
            reset_callback (function): Scenario reset function
            reward_callback (function): Scenario reward function
            observation_callback (function): Scenario observation function
            logging_callback (function): Scenario logging function
            info_callback (function): Scenario benchmark info function
            done_callback (function): Scenario done function
            shared_viewer: (boolean) Specifies whether to share a viewer window or create one for each agent
        """

        # Set the world and policy agents
        self.world = world
        self.arglist = arglist
        self.logger = logger
        self.agents = self.world.policy_agents

        # Set required vectorized gym env property
        self.n = len(world.policy_agents)

        # Scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.logging_callback = logging_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # Environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector

        self.discrete_action_input = world.discrete_action_input if hasattr(world, 'discrete_action_input') else False
        # if true, even the action is continuous, action will be performed discretely

        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False

        # Separate capture action
        self.capture_action = world.capture_action if hasattr(world, 'capture_action') else False

        # Separate capture actions when catching together
        self.num_capture_actions = world.num_capture_actions if hasattr(world, 'num_capture_actions') else 1

        # Check for turret scenario
        self.turret_scenario = self.world.turret if hasattr(self.world, 'turret') else False
        self.turret_cg_scenario = self.world.turret_cg if hasattr(self.world, 'turret_cg') else False

        # Check for stag hunt scenario
        self.stag_hunt_scenario = self.world.stag_hunt if hasattr(self.world, 'stag_hunt') else False

        self.time = 0

        # Configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []

            # Physical action space, either it is discrete or continuous.
            # Only movable agents have a physical action space
            if agent.movable or isinstance(agent, TurretAgent):
                if self.discrete_action_space:
                    # Example: A world dimension of 2 allows an agent to move using x,y coordinates.
                    #          In this world the agent has the following actions: No Action, Left, Right, Forward, Back.
                    #          This gives a discrete space with 5 actions.
                    if not self.capture_action:
                        u_action_space = gym.spaces.Discrete(world.dimension_position * 2 + 1)
                    else:
                        u_action_space = gym.spaces.Discrete(world.dimension_position * 2 + 1 +
                                                             self.num_capture_actions)

                else:
                    u_action_space = gym.spaces.Box(low=-agent.u_range, high=+agent.u_range,
                                                    shape=(world.dimension_position,), dtype=np.float32)

                # Add physical action space to total action space
                total_action_space.append(u_action_space)

            # Communication action space, either it is discrete or continuous.
            # Only non-silent agents have a communication action space
            if not agent.silent:
                if self.discrete_action_space:
                    c_action_space = gym.spaces.Discrete(world.dimension_communication)
                else:
                    c_action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(world.dimension_communication,),
                                                    dtype=np.float32)

                # Add communication action space to total action space
                total_action_space.append(c_action_space)

            # Total action space
            if len(total_action_space) > 1:
                # All action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, gym.spaces.Discrete) for act_space in total_action_space]):
                    act_space = gym.spaces.multi_discrete.MultiDiscrete([act_space.n for
                                                                         act_space in total_action_space])
                else:
                    act_space = gym.spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # Observation space
            if not self.stag_hunt_scenario and not self.turret_cg_scenario:
                obs_dim = len(observation_callback(agent, self.world))
            else:
                obs_dim = self.world.scenario.agent_obs_size
            self.observation_space.append(gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dimension_communication)

        # Rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

        # Set up logging
        self.logger.new("State",
                        ["Episode", "Step", "Agent"] + world.log_headers + ["Reward", "Done", "Info"])

    def step(self, action_n, dense_reward=False):
        """
        Advance the environment a step

        Args:
            action_n (list): Actions for n-number of agents

        Returns:
            obs_n (list): Observations for n-number of agents
            reward_n (list): Rewards for n-number of agents
            done_n (list): Dones for n-number of agents
            info_n (dictionary): Benchmarking info for n-number of agents
        """
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents

        # Set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        # Advance world state
        self.world.step()

        # Record observation for each agent
        #
        # This was changed so that observations are reported for all agents,
        # not just learning agents. This is necessary if we want learning agents
        # to be able to learn from the actions of fixed policy agents.
        #
        # for agent in self.agents:
        if not self.stag_hunt_scenario and not self.turret_cg_scenario:
            for agent in self.world.agents:
                obs_n.append(self._get_obs(agent))
                reward_n.append(self._get_reward(agent, dense_reward))
                done_n.append(self._get_done(agent))
                info_n['n'].append(self._get_info(agent))

            # All agents get total reward in cooperative case
            if self.shared_reward:
                reward = np.sum(reward_n)
                reward_n = [reward] * self.n

            if self.turret_scenario:
                # Purely for color changing, not required;
                # should be very quick even if not displaying anything
                self._reset_render()

            return obs_n, reward_n, done_n, info_n

        else:
            if self.world.scenario.use_adj_matrix:
                obs_n, adjacency_matrix, available_actions_n, state = self._get_obs()

                done = self._get_done()
                info = self._get_info()
                reward = self._get_reward()

                # Purely for color changing, not required;
                # should be very quick even if not displaying anything
                self._reset_render()

                return obs_n, adjacency_matrix, available_actions_n, state, reward, done, info

            elif self.world.scenario.use_hyperedge_index:
                obs_n, hyperedge_index, available_actions_n, state = self._get_obs()

                done = self._get_done()
                info = self._get_info()
                reward = self._get_reward()

                # Purely for color changing, not required;
                # should be very quick even if not displaying anything
                self._reset_render()

                return obs_n, hyperedge_index, available_actions_n, state, reward, done, info

            else:
                obs_n, available_actions_n, state = self._get_obs()

                done = self._get_done()
                info = self._get_info()
                reward = self._get_reward()

                # Purely for color changing, not required;
                # should be very quick even if not displaying anything
                self._reset_render()

                return obs_n, available_actions_n, state, reward, done, info

    def log(self, episode, step, observations, reward, done, info):
        """
        Log data for n-number of agents at a given episode and step.

        Args:
            episode (int):                    Current episode
            step (int):                       Current episode step
            observations (list):              Observations for n-number of agents for a given episode and step
            reward (list/np.array/float/int): Rewards for n-number of agents for a given episode and step
            done (list/bool):                 Dones for n-number of agents for a given episode and step
            info (list/dict):                 Benchmarking info for n-number of agents for a given episode and step
        """
        if not self.stag_hunt_scenario and not self.turret_cg_scenario:
            for i, agent in enumerate(self.world.agents):
                agent_data = [episode, step, i] + self.logging_callback(agent, self.world) + [reward[i],
                                                                                              done[i],
                                                                                              info['n'][i]]
                self.logger.add("State", agent_data)
        else:
            for i, agent in enumerate(self.world.agents):
                agent_data = [episode, step, i] + self.logging_callback(agent, self.world) + [reward, done, info]
                self.logger.add("State", agent_data)

            for i, agent in enumerate(self.world.random_agents):
                agent_data = [episode, step, i] + self.logging_callback(agent, self.world) + [reward, done, info]
                self.logger.add("State", agent_data)

    def reset(self):
        """
        Reset the environment

        Returns:
            obs_n (list): Observations for n-number of agents
        """
        # Reset world
        self.reset_callback(self.world)

        # Reset renderer
        self._reset_render()

        if not self.stag_hunt_scenario and not self.turret_cg_scenario:
            # Record observations for each agent
            obs_n = []
            self.agents = self.world.policy_agents
            for agent in self.agents:
                obs_n.append(self._get_obs(agent))

            return obs_n
        else:
            # Record observations and available actions for each agent and the grid state
            if self.world.scenario.use_adj_matrix:
                obs_n, adjacency_matrix, available_actions_n, state = self._get_obs()

                return obs_n, adjacency_matrix, available_actions_n, state

            # TODO: Remove if not needed anymore
            # elif self.world.scenario.use_hyperedge_index:
            #     obs_n, hyperedge_index, available_actions_n, state = self._get_obs()
            #
            #     return obs_n, hyperedge_index, available_actions_n, state

            else:
                obs_n, available_actions_n, state = self._get_obs()

                return obs_n, available_actions_n, state

    def _get_info(self, agent=None):
        """
        Returns info used for benchmarking

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object

        Returns:
            A dictionary containing benchmarking info for a given agent

        """
        if self.info_callback is None:
            return {}

        if agent is not None:
            return self.info_callback(agent, self.world)
        else:
            return self.info_callback(self.world)

    def _get_obs(self, agent=None):
        """
        Returns observations for a particular agent

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object

        Returns:
            A  np.array containing the world observations of a given agent
        """
        if self.observation_callback is None:
            return np.zeros(0)

        if agent is not None:
            return self.observation_callback(agent, self.world)
        else:
            return self.observation_callback(self.world)

    def _get_done(self, agent=None):
        """
        Returns dones for a particular agent

        Unused right now -- agents are allowed to go beyond the viewing screen

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object

        Returns:
            A boolean which specifies whether a given agent is done
        """
        if self.done_callback is None:
            return False

        if agent is not None:
            return self.done_callback(agent, self.world)
        else:
            return self.done_callback(self.world)

    def _get_reward(self, agent=None, dense_reward=False):
        """
        Return reward for a particular agent

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object

        Returns:
            A float which is the reward for an action taken by a given agent
        """
        if self.reward_callback is None:
            return 0.0

        if dense_reward and agent is not None:
            return self.reward_callback(agent, self.world, dense=dense_reward)
        elif agent is not None:
            return self.reward_callback(agent, self.world)
        elif dense_reward:
            return self.reward_callback(self.world, dense=dense_reward)
        else:
            return self.reward_callback(self.world)

    def _set_action(self, action, agent, action_space):
        """
        Set action for a particular agent

        Args:
            action ():
            agent (multiagent_particle_env.core.Agent): Agent object
            action_space (gym.spaces.Discrete/multi_discrete.MultiDiscrete/Box/Tuple): Action space of the agent
                          If actions are discrete and agent has no communication
                          (gym.spaces.Discrete)
                          If actions are discrete and agent has communication
                          (gym.spaces.multi_discrete.MultiDiscrete)
                          If actions are not discrete and agent has no communication
                          (gym.spaces.Box)
                          If actions are not discrete and agent has communication
                          gym.spaces.Tuple(gym.spaces.Box, gym.spaces.Box)
        """
        agent.action.u = np.zeros(self.world.dimension_position)
        agent.action.c = np.zeros(self.world.dimension_communication)
        # Process action
        if isinstance(action_space, gym.spaces.multi_discrete.MultiDiscrete):
            act = []
            size = action_space.nvec
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        # Physical action
        if agent.movable or isinstance(agent, TurretAgent):
            if self.discrete_action_input:
                # Process discrete action
                agent.action.u = np.zeros(self.world.dimension_position)
                agent.action.capture = np.zeros(self.num_capture_actions, dtype=bool)
                agent.action.discrete_raw = action[0]

                # Stay or Do Nothing Action
                if action[0] == 0:
                    pass

                # Move Left
                elif action[0] == 1:
                    agent.action.u[0] = -1.0

                # Move Right
                elif action[0] == 2:
                    agent.action.u[0] = +1.0

                # Move Down
                elif action[0] == 3:
                    agent.action.u[1] = -1.0

                # Move Up
                elif action[0] == 4:
                    agent.action.u[1] = +1.0

                # Capture action
                elif self.capture_action and (action[0] >= 5) and (action[0] <= (5 + (self.num_capture_actions - 1))):
                    agent.action.capture[(action[0] - 5)] = True

                else:
                    raise ValueError(f"Discrete action value: {action[0]} not supported")
            else:
                # Process forced discrete action
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0

                if self.discrete_action_space:
                    # Process discrete action space action
                    # print("THIS IS WHAT IS HAPPENING")
                    # print("Actions: {}".format(action))

                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    # Process continuous action
                    agent.action.u = action[0]

            # Apply acceleration to agent action or use default
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

            # Step to next action element
            action = action[1:]

        # Communication action
        if not agent.silent:
            if self.discrete_action_input:
                # Process discrete action
                agent.action.c = np.zeros(self.world.dimension_communication)
                agent.action.c[action[0]] = 1.0
            else:
                # Process continuous action
                agent.action.c = action[0]

            # Step to next action element
            action = action[1:]

        # Ensure we used all elements of action
        assert len(action) == 0

    def _reset_render(self):
        """
        Reset rendering assets
        """
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human', save_filename=None):
        """
        Render environment

        Args:
            mode (string):

        Returns:
            results: ()
        """
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # Create viewers (if necessary)
            if self.viewers[i] is None:
                # Import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent_particle_env import rendering

                self.viewers[i] = rendering.Viewer(700, 700)

        # Create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent_particle_env import rendering

            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                if entity.has_sense and not isinstance(entity, TurretAgent):
                    geom = rendering.make_circle(entity.size)
                    geom_sense = rendering.make_circle(radius=entity.size + entity.sense_region, filled=False)

                    xform = rendering.Transform()
                    if 'agent' in entity.name:
                        # Unpack color list
                        if len(entity.color) == 3:
                            geom.set_color(*entity.color, alpha=0.5)
                            geom_sense.set_color(*entity.color, alpha=0.5)
                        else:
                            geom.set_color(*entity.color[:3], alpha=entity.color[3])
                            geom_sense.set_color(*entity.color[:3], alpha=entity.color[3])
                    else:
                        # Unpack color list
                        if len(entity.color) == 3:
                            geom.set_color(*entity.color)
                            geom_sense.set_color(*entity.color)
                        else:
                            geom.set_color(*entity.color[:3], alpha=entity.color[3])
                            geom_sense.set_color(*entity.color[:3], alpha=entity.color[3])
                    geom.add_attr(xform)
                    geom_sense.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms.append(geom_sense)
                    self.render_geoms_xform.append(xform)
                elif entity.has_sense and isinstance(entity, TurretAgent):
                    # Turret body
                    geom = rendering.make_circle(entity.size)
                    geom.set_color(*entity.color)

                    # Turret sense region
                    # changed sense region to be from center not edge of agent
                    geom_sense = rendering.make_circle(radius=entity.sense_region, filled=False)
                    geom_sense.set_color(*entity.color)

                    # Turret goal region
                    geom_goal_region = rendering.make_circle(radius=entity.goal_radius, filled=False)
                    geom_goal_region.set_color(r=1, g=.65, b=0)

                    # Turret body transform
                    xform_body = rendering.Transform()
                    geom.add_attr(xform_body)
                    geom_sense.add_attr(xform_body)
                    geom_goal_region.add_attr(xform_body)

                    # Turret Head
                    geom_head = rendering.make_circle(entity.head_size)
                    geom_head.set_color(*entity.head_color)

                    # Turret Head Line
                    turret_line_len = .75
                    # entity is a length of .1, divide by .1 to get 1 then multiply by desired length
                    geom_head_line = rendering.Line(end=(entity.state.p_head_pos[0] / .1 * turret_line_len,
                                                         entity.state.p_head_pos[1] / .1 * turret_line_len))
                    geom_head_line.set_color(r=.25, g=.25, b=.25)

                    xform_head = rendering.Transform()
                    geom_head.add_attr(xform_head)
                    geom_head_line.add_attr(xform_head)

                    self.render_geoms.append(geom)
                    self.render_geoms.append(geom_sense)
                    self.render_geoms.append(geom_goal_region)
                    self.render_geoms.append(geom_head)
                    self.render_geoms.append(geom_head_line)
                    self.render_geoms_xform.append([xform_body, xform_head])
                else:
                    geom = rendering.make_circle(entity.size)
                    xform = rendering.Transform()
                    if 'agent' in entity.name:
                        # Unpack color list
                        if len(entity.color) == 3:
                            geom.set_color(*entity.color, alpha=0.5)
                        else:
                            geom.set_color(*entity.color[:3], alpha=entity.color[3])
                    else:
                        # Unpack color list
                        if len(entity.color) == 3:
                            geom.set_color(*entity.color)
                        else:
                            geom.set_color(*entity.color[:3], alpha=entity.color[3])
                    geom.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)

            # Add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent_particle_env import rendering

            # Update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dimension_position)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)

            # Update geometry positions
            for e, entity in enumerate(self.world.entities):
                if not isinstance(entity, TurretAgent):
                    self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                else:
                    self.render_geoms_xform[e][0].set_translation(*entity.state.p_pos)
                    self.render_geoms_xform[e][1].set_translation(*entity.state.p_head_pos)

            # Render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array', save_filename=save_filename))

        return results
