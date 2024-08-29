# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_adversary.py

Physical Deception

Communication: No
Competitive:   Yes

1 adversary (red), N good agents (green), N landmarks (usually N=2).

All agents observe position of landmarks and other agents.
One landmark is the ‘target landmark’ (colored green).

Good agents rewarded based on how close one of them is to the target landmark,
but negatively rewarded if the adversary is close to target landmark.

Adversary is rewarded based on how close it is to the target,
but it doesn’t know which landmark is the target landmark.

So good agents have to learn to ‘split up’ and cover all landmarks to deceive the adversary.

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
        num_agents = 3
        num_landmarks = num_agents - 1

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.adversary = True if i < num_adversaries else False
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark {}'.format(i)
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08

        # Make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Set properties for agents
        for agent in world.agents:
            if agent.adversary:
                agent.color = np.array([0.85, 0.35, 0.35])
            else:
                agent.color = np.array([0.35, 0.35, 0.85])

        # Set properties for landmarks
        for landmark in world.landmarks:
            landmark.color = np.array([0.15, 0.15, 0.15])

        # Set goal landmark
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.c = np.zeros(world.dimension_communication)
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)

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
            If the agent is the adversary:
                (float) The distance squared between the agent and the goal landmark
            Else:
                (tuple) The distance squared between the agent and all landmarks
                and between the agent and it's goal landmark
        """
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []

            # Collect distances between the agent and all the landmarks
            for landmark in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))

            # Collect distance between the agent and it's goal landmark
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))

            # Convert list to tuple
            return tuple(dists)

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

        Returns:
            (list) All the agents in the world that are adversaries.
        """
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world, shaped_reward=True, shaped_adv_reward=True):
        """
        Good Agents are rewarded based on how close any good agent is to the goal landmark,
        and how far the adversary is from it.

        Adversary Agents are rewarded based on distance/proximity to the goal landmark.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shaped_reward (boolean): Specifies whether to use shaped reward, distance based vs proximity based
            shaped_adv_reward (boolean): Specifies whether to use shaped adversary reward in good agent reward,
                                         distance based vs proximity based

        Returns:
            If agent is adversary:
                self.adversary_reward() result
            Else:
                self.agent_reward() result
        """
        if agent.adversary:
            return self.adversary_reward(agent, shaped_reward)
        else:
            return self.agent_reward(agent, world, shaped_reward, shaped_adv_reward)

    def agent_reward(self, agent, world, shaped_reward, shaped_adv_reward):
        """
        Reward based on how close any good agent is to the goal landmark,
        and how far the adversary is from it.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shaped_reward (boolean): Specifies whether to use shaped reward, distance based vs proximity based
            shaped_adv_reward (boolean): Specifies whether to use shaped adversary reward,
                                         distance based vs proximity based

        Returns:
            (float) Total reward, positive good agent reward plus negative adversary agent reward
        """
        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:
            # Distance-based adversary reward
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:
            # Proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:
            # Distance-based agent reward
            pos_rew = -min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        else:
            # Proximity-based agent reward (binary)
            pos_rew = 0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                    < 2 * agent.goal_a.size:
                pos_rew += 5

            pos_rew -= min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])

        return pos_rew + adv_rew

    def adversary_reward(self, agent, shaped_reward):
        """
        Reward based on distance/proximity to the goal landmark.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            shaped_reward (boolean): Specifies whether to use shaped reward, distance based vs proximity based

        Returns:
            (float) Adversarial agent reward
        """
        if shaped_reward:
            # Distance-based reward, squared distance
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            # Proximity-based reward (binary)
            adv_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew += 5

            return adv_rew

    def observation(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            If agent is not an adversary:
                (np.array) Observations array with distance to goal landmark, distance to all landmarks,
                           and distance to all other agents in the agent's reference frame.
            Else:
                (np.array) Observations array with distance to distance to all landmarks
                           and distance to all other agents in the agent's reference frame.
        """
        # Get positions of all landmarks in this agent's reference frame
        landmarks_pos = []
        for landmark in world.landmarks:
            landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)

        # Get positions of all other agents in this agent's reference frame
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + landmarks_pos + other_pos)
        else:
            return np.concatenate(landmarks_pos + other_pos)
