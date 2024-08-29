# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_world_comm.py

Predator-prey

Communication: Yes
Competitive:   Yes

6 agents, 5 landmarks.

Environment seen in the video accompanying the paper.

Same as simple_tag, except for the following:
    (1) There is food (small blue balls) that the good agents are rewarded for being near,
    (2) We now have ‘forests’ that hide agents inside from being seen from outside;
    (3) There is a ‘leader adversary” that can see the agents at all times, and can
        communicate with the other adversaries to help coordinate the chase.

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
        # world.damping = 1
        world.dimension_communication = 4
        num_good_agents = 2
        num_adversaries = 4
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 1
        num_food = 2
        num_forests = 2

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.adversary = True if i < num_adversaries else False
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.collide = True
            agent.leader = True if i == 0 else False
            agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.silent = True if i > 0 else False
            agent.size = 0.075 if agent.adversary else 0.045

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark {}'.format(i)
            landmark.boundary = False
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2

        # Add food landmarks
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food {}'.format(i)
            landmark.boundary = False
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
        world.landmarks += world.food

        # Add forest landmarks
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
            landmark.name = 'forest {}'.format(i)
            landmark.boundary = False
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.3
        world.landmarks += world.forests

        # World boundaries now penalized with negative reward
        # world.landmarks += world.set_boundaries()

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
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])

        # Set properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # Set properties for food landmarks
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])

        # Set properties for forest landmarks
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

        # Set random initial states for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dimension_position)
            landmark.state.p_vel = np.zeros(world.dimension_position)

        # Set random initial states for food landmarks
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dimension_position)
            landmark.state.p_vel = np.zeros(world.dimension_position)

        # Set random initial states for forest landmarks
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dimension_position)
            landmark.state.p_vel = np.zeros(world.dimension_position)

    def benchmark_data(self, agent, world):
        """
        Returns data for benchmarking purposes.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            If agent is an adversary:
                (int) The total number of collisions between all good agents
            Else:
                (int) Zero
        """
        if agent.adversary:
            # Determine total number of collisions with all good agents
            collisions = 0
            for a in self.good_agents(world):
                if world.is_collision(a, agent):
                    collisions += 1

            return collisions
        else:
            return 0

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

    def reward(self, agent, world, shaped=False, shaped_adv=True):
        """
        Reward is based on prey agent not being caught by predator agents.

        Good agents are negatively rewarded if caught by adversaries and for exiting the screen.
        They are positively reward for colliding with and being near food landmarks.

        Adversaries are rewarded for collisions with good agents.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shaped (boolean): Specifies whether to use shaped reward, adds distance based increase
            shaped_adv (boolean): Specifies whether to use shaped reward, adds distance based decrease

        Returns:
            If agent is adversary:
                self.adversary_reward() result
            Else:
                self.agent_reward() result
        """
        if agent.adversary:
            return self.adversary_reward(agent, world, shaped_adv)
        else:
            return self.agent_reward(agent, world, shaped)

    def outside_boundary(self, agent):
        """

        """
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or \
                agent.state.p_pos[1] < -1:
            return True
        else:
            return False

    def agent_reward(self, agent, world, shaped):
        """
        Good agents are negatively rewarded if caught by adversaries and for exiting the screen.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shaped (boolean): Specifies whether to use shaped reward, increased for increased distance from adversaries.

        Returns:
            (float) Total agent reward, based on avoiding collisions and staying within the screen.
        """
        reward = 0
        adversaries = self.adversaries(world)

        # Reward can optionally be shaped (increased reward for increased distance from adversary)
        if shaped:
            for adv in adversaries:
                reward += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))

        # Determine collisions and assign penalties
        if agent.collide:
            for adv in adversaries:
                if world.is_collision(adv, agent):
                    reward -= 5

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            reward -= 2 * world.bound(abs(agent.state.p_pos[coordinate_position]))

        # Determine reward for collisions with food landmarks
        for food in world.food:
            if world.is_collision(agent, food):
                reward += 2

        # Determine reward for being near food landmarks
        reward += 0.05 * min([np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos))) for food in world.food])

        return reward

    def adversary_reward(self, agent, world, shaped_adv):
        """
        Adversaries are rewarded for collisions with good agents.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            shaped_adv (boolean): Specifies whether to use shaped reward,
                                  decreased for increased distance from good agents.

        Returns:
            (float) Total agent reward, based on avoiding collisions and staying within the screen.
        """
        # Adversaries are rewarded for collisions with agents
        reward = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        # Reward can optionally be shaped (decreased reward for increased distance from agents)
        if shaped_adv:
            reward -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents])

        # Determine collisions and assign rewards
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if world.is_collision(ag, adv):
                        reward += 5

        return reward

    def observation2(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (np.array) Observations array with the velocity of the agent, the position of the agent,
                       distance to all landmarks in the agent's reference frame,
                       distance to all other agents in the agent's reference frame,
                       and the velocities of the good agents.
        """
        # Get positions all landmarks that are not boundary markers in this agent's reference frame
        landmarks_pos = []
        for landmark in world.landmarks:
            if not landmark.boundary:
                landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)

        # Get positions all food landmarks that are not boundary markers in this agent's reference frame
        food_pos = []
        for food in world.food:
            if not food.boundary:
                food_pos.append(food.state.p_pos - agent.state.p_pos)

        # Communication, positions, and velocities of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

            # Only store velocities for good agents
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmarks_pos + other_pos + other_vel)

    def observation(self, agent, world):
        """
        Define the observations.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            If agent is adversary and not leader:
                (np.array) Observations array with the velocity of the agent, the position of the agent,
                           distance to all landmarks in the agent's reference frame,
                           distance to all other agents in the agent's reference frame,
                           the velocities of the good agents,
                           whether the agent is in the forest,
                           and the communication states of the other agents.
            Elif agent is leader:
                (np.array) Observations array with the velocity of the agent, the position of the agent,
                           distance to all landmarks in the agent's reference frame,
                           distance to all other agents in the agent's reference frame,
                           the velocities of the good agents,
                           whether the agent is in the forest,
                           and the communication states of the other agents.
            Else:
                (np.array) Observations array with the velocity of the agent, the position of the agent,
                           distance to all landmarks in the agent's reference frame,
                           distance to all other agents in the agent's reference frame,
                           whether the agent is in the forest,
                           and the velocities of the good agents.
        """
        # Get positions all landmarks that are not boundary markers in this agent's reference frame
        landmarks_pos = []
        for landmark in world.landmarks:
            if not landmark.boundary:
                landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)

        # Get positions all food landmarks that are not boundary markers in this agent's reference frame
        food_pos = []
        for food in world.food:
            if not food.boundary:
                food_pos.append(food.state.p_pos - agent.state.p_pos)

        # Determine if agent is in the forest
        in_forest = [np.array([-1]), np.array([-1])]
        inf1 = False
        inf2 = False
        if world.is_collision(agent, world.forests[0]):
            in_forest[0] = np.array([1])
            inf1 = True
        if world.is_collision(agent, world.forests[1]):
            in_forest[1] = np.array([1])
            inf2 = True

        # Communication, positions, and velocities of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)

            # Determine if other agent is in the forest
            oth_f1 = world.is_collision(other, world.forests[0])
            oth_f2 = world.is_collision(other, world.forests[1])
            if (inf1 and oth_f1) or (inf2 and oth_f2) or (not inf1 and not oth_f1 and not inf2 and not oth_f2) or \
                    agent.leader:
                other_pos.append(other.state.p_pos - agent.state.p_pos)

                # Only store velocities for good agents
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            else:
                other_pos.append([0, 0])

                # Only store velocities for good agents
                if not other.adversary:
                    other_vel.append([0, 0])

        # Tell the predator when the prey are in the forest
        prey_forest = []
        good_agents = self.good_agents(world)
        for ga in good_agents:
            if any([world.is_collision(ga, forest) for forest in world.forests]):
                prey_forest.append(np.array([1]))
            else:
                prey_forest.append(np.array([-1]))

        # Tell the leader when the prey are in forest
        prey_forest_lead = []
        for forest in world.forests:
            if any([world.is_collision(ga, forest) for ga in good_agents]):
                prey_forest_lead.append(np.array([1]))
            else:
                prey_forest_lead.append(np.array([-1]))

        comm = [world.agents[0].state.c]

        if agent.adversary and not agent.leader:
            return np.concatenate([agent.state.p_vel] +
                                  [agent.state.p_pos] +
                                  landmarks_pos +
                                  other_pos +
                                  other_vel +
                                  in_forest +
                                  comm)

        elif agent.leader:
            return np.concatenate([agent.state.p_vel] +
                                  [agent.state.p_pos] +
                                  landmarks_pos +
                                  other_pos +
                                  other_vel +
                                  in_forest +
                                  comm)
        else:
            return np.concatenate([agent.state.p_vel] +
                                  [agent.state.p_pos] +
                                  landmarks_pos +
                                  other_pos +
                                  in_forest +
                                  other_vel)
