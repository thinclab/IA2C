# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_crypto.py

Covert Communication

Communication: Yes
Competitive:   Yes

Two good agents (alice and bob), one adversary (eve).
Alice must sent a private message to bob over a public channel.

Alice and bob are rewarded based on how well bob reconstructs the message,
but negatively rewarded if eve can reconstruct the message.

Alice and bob have a private key (randomly generated at beginning of each episode),
which they must learn to use to encrypt the message.

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


class CryptoAgent(Agent):
    """
    Properties of crypto agent Entities
    """

    def __init__(self):
        super(CryptoAgent, self).__init__()
        self.key = None


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
        world.dimension_communication = 4
        num_adversaries = 1
        num_agents = 3
        num_landmarks = 2

        # Add agents
        world.agents = [CryptoAgent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.adversary = True if i < num_adversaries else False
            agent.collide = False
            agent.movable = False
            agent.speaker = True if i == 2 else False

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark {}'.format(i)
            landmark.collide = False
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
        # Set properties for agents
        for agent in world.agents:
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                agent.color = np.array([0.25, 0.25, 0.25])
            agent.key = None

        # Set properties for landmarks
        color_list = [np.zeros(world.dimension_communication) for i in world.landmarks]
        for i, color in enumerate(color_list):
            color[i] += 1
        for color, landmark in zip(color_list, world.landmarks):
            landmark.color = color

        # Set goal landmark
        goal = np.random.choice(world.landmarks)
        world.agents[1].color = goal.color
        world.agents[2].key = np.random.choice(world.landmarks).color
        for agent in world.agents:
            agent.goal_a = goal

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

        # Set random initial states for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            landmark.state.p_vel = np.zeros(world.dimension_position)

    def benchmark_data(self, agent, world):
        """
        Returns data for benchmarking purposes.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (tuple) The agent's communication state and goal color
        """
        return tuple([agent.state.c, agent.goal_a.color])

    def good_listeners(self, world):
        """
        Returns all agents that are not adversaries and are not speakers in a list.

        Returns:
            (list) All the agents in the world that are not adversaries.
        """
        return [agent for agent in world.agents if not agent.adversary and not agent.speaker]

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

    def reward(self, agent, world):
        """
        Good Agents are rewarded if Bob can reconstruct message, but adversary (Eve) cannot.

        Adversary Agent (Eve) is rewarded  if it can reconstruct original goal.

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
        Reward based on if Bob can reconstruct message, but adversary (Eve) cannot.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (float) Total reward, Bob's reconstruction plus Eve's reconstruction
        """
        good_listeners = self.good_listeners(world)
        adversaries = self.adversaries(world)
        good_rew = 0
        adv_rew = 0

        # Calculate Bob's reconstruction reward
        for a in good_listeners:
            if (a.state.c == np.zeros(world.dimension_communication)).all():
                continue
            else:
                good_rew -= np.sum(np.square(a.state.c - agent.goal_a.color))

        # Calculate Eve's reconstruction reward
        for a in adversaries:
            if (a.state.c == np.zeros(world.dimension_communication)).all():
                continue
            else:
                adv_rew += np.sum(np.square(a.state.c - agent.goal_a.color))

        return adv_rew + good_rew

    def adversary_reward(self, agent, world):
        """
        Reward based on if Eve can reconstruct original goal

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (float) Eve's reconstruction
        """
        reward = 0

        # Calculate Eve's reconstruction reward
        if not (agent.state.c == np.zeros(world.dimension_communication)).all():
            reward -= np.sum(np.square(agent.state.c - agent.goal_a.color))

        return reward

    def observation(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            If agent is speaker: (Speaker)
                (np.array) Observations array with goal color and key
            If agent is not speaker and is not adversary: (Listener)
                (np.array) Observations array with key and other agents communication state.
            If agent is not speaker and is adversary: (Adversary)
                (np.array) Observations array with other agents communication state.
        """
        # Goal color
        goal_color = np.zeros(world.dimension_color)
        if agent.goal_a is not None:
            goal_color = agent.goal_a.color

        # Get positions of all landmarks in this agent's reference frame
        landmarks_pos = []
        for landmark in world.landmarks:
            landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)

        # Communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None) or not other.speaker:
                continue
            comm.append(other.state.c)

        confer = np.array([0])
        if world.agents[2].key is None:
            confer = np.array([1])
            key = np.zeros(world.dimension_communication)
            goal_color = np.zeros(world.dimension_communication)
        else:
            key = world.agents[2].key

        debug = False
        # Speaker
        if agent.speaker:
            if debug:
                print('speaker')
                print(agent.state.c)
                print(np.concatenate([goal_color] + [key] + [confer] + [np.random.randn(1)]))
            return np.concatenate([goal_color] + [key])

        # Listener
        if not agent.speaker and not agent.adversary:
            if debug:
                print('listener')
                print(agent.state.c)
                print(np.concatenate([key] + comm + [confer]))
            return np.concatenate([key] + comm)

        # Adversary
        if not agent.speaker and agent.adversary:
            if debug:
                print('adversary')
                print(agent.state.c)
                print(np.concatenate(comm + [confer]))
            return np.concatenate(comm)
