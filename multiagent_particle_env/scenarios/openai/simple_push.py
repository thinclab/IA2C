# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_push.py

Keep Away

Communication: No
Competitive:   Yes

1 agent, 1 adversary, 1 landmark.

Agent is rewarded based on distance to landmark.

Adversary is rewarded if it is close to the landmark, and if the agent is far from the landmark.
So the adversary learns to push agent away from the landmark.

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
        world.dimension_communication = 2
        num_adversaries = 1
        num_agents = 2
        num_landmarks = 2

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.adversary = True if i < num_adversaries else False
            agent.collide = True
            agent.index = i
            agent.silent = True

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark {}'.format(i)
            landmark.collide = False
            landmark.index = i
            landmark.movable = False

        # Make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Set properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[i + 1] += 0.8

        # Set goal landmark
        goal = np.random.choice(world.landmarks)
        for agent in world.agents:
            agent.goal_a = goal
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                agent.color = np.array([0.25, 0.25, 0.25])
                agent.color[goal.index + 1] += 0.5

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

        # Set random initial states for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            landmark.state.p_vel = np.zeros(world.dimension_position)

    def reward(self, agent, world):
        """
        Good Agents are rewarded based on minimum agent distance to each landmark.

        Adversary Agents are rewarded based if they are close to the landmark,
        and if the agent is far from the landmark. So the adversary learns to push agent away from the landmark.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            If agent is adversary:
                self.adversary_reward() result
            Else:
                self.agent_reward() result
        """
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        """
        Rewarded based on minimum agent distance to goal landmark.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (float) Good agent reward
        """
        # Negative distance
        return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))

    def adversary_reward(self, agent, world):
        """
        Reward based on if it is close to the landmark,
        and if the agent is far from the landmark. So the adversary learns to push agent away from the landmark.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (float) Total reward, positive reward minus negative reward
        """
        # Keep the good agents away from the goal
        agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos -
                                               a.goal_a.state.p_pos))) for a in world.agents if not a.adversary]
        positive_reward = min(agent_dist)

        # Reward based on distance to nearest good agent
        # nearest_agent = world.good_agents[np.argmin(agent_dist)]
        # negative_reward = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))

        # Reward based on distance to every good agent
        # negative_reward = sum([np.sqrt(np.sum(np.square(a.state.p_pos -
        #                                                 agent.state.p_pos))) for a in world.good_agents])

        # Reward based on distance to goal
        negative_reward = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))

        return positive_reward - negative_reward
               
    def observation(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            If agent is not an adversary:
                (np.array) Observations array with the velocity of the agent,
                           distance to goal in the agent's reference frame,
                           color of the agent,
                           distance to all landmarks in the agent's reference frame,
                           colors of all landmarks,
                           and distance to all other agents in the agent's reference frame.
            Else:
                (np.array) Observations array with the velocity of the agent,
                           distance to all landmarks in the agent's reference frame,
                           and distance to all other agents in the agent's reference frame.
        """
        # Get positions and colors of all landmarks in this agent's reference frame
        landmarks_pos = []
        landmarks_color = []
        for landmark in world.landmarks:
            landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)
            landmarks_color.append(landmark.color)

        # Communications and positions of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate([agent.state.p_vel] +
                                  [agent.goal_a.state.p_pos - agent.state.p_pos] +
                                  [agent.color] +
                                  landmarks_pos +
                                  landmarks_color +
                                  other_pos)
        else:
            # randomize position of other agents in adversary network
            # other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos
            return np.concatenate([agent.state.p_vel] +
                                  landmarks_pos +
                                  other_pos)
