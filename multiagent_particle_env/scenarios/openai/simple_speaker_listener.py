# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_speaker_listener.py

Cooperative Communication

Communication: Yes
Competitive:   No

2 agents, 3 landmarks of different colors.

Same as simple_reference, except one agent is the ‘speaker’ (gray) that does not move (observes goal of other agent),
and other agent is the listener (cannot speak, but must navigate to correct landmark).

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import numpy as np

from multiagent_particle_env.core import World, Agent, Landmark
from multiagent_particle_env.scenario import BaseScenario

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

    def make_world(self, args=None):
        """
        Construct the world

        Returns:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Create world and set properties
        world = World()
        world.collaborative = True
        world.dimension_communication = 3
        num_landmarks = 3

        # Add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.collide = False
            agent.size = 0.075

        # Set speaker
        world.agents[0].movable = False

        # Set listener
        world.agents[1].silent = True

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark {}'.format(i)
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04

        # Make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)

        # Set properties for agents
        for agent in world.agents:
            agent.color = np.array([0.25, 0.25, 0.25])

        # Set properties for landmarks
        world.landmarks[0].color = np.array([0.65, 0.15, 0.15])
        world.landmarks[1].color = np.array([0.15, 0.65, 0.15])
        world.landmarks[2].color = np.array([0.15, 0.15, 0.65])

        # Special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

        # Set random initial states for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            landmark.state.p_vel = np.zeros(world.dimension_position)

    def benchmark_data(self, agent, world):
        """
        Returns data for benchmarking purposes.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (float) The reward for the agent
        """
        return self.reward(agent, world)

    def reward(self, agent, world):
        """
        Reward is collective. So listener has to learn to listen for the goal from the speaker,
        and navigate to their landmark.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (float) Negative distance squared of the other agent to their goal landmark
        """
        # Negative distance squared from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))

        return -dist2

    def observation(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            If agent is not movable: (Speaker)
                (np.array) Observations array with the color of the goal.
            Else: (Listener)
                (np.array) Observations array with the velocity of the agent,
                           distance to all landmarks in the agent's reference frame,
                           and the communication state of all other agents.
        """
        # Goal color
        goal_color = np.zeros(world.dimension_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # Get positions of all landmarks in this agent's reference frame
        landmarks_pos = []
        for landmark in world.landmarks:
            landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)

        # Communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            comm.append(other.state.c)
        
        # Speaker
        if not agent.movable:
            return np.concatenate([goal_color])

        # Listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + landmarks_pos + comm)
