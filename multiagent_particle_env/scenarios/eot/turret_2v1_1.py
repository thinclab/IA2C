# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
turret_2v1.py

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import numpy as np

from multiagent_particle_env.alternate_policies import fixed_turret
from multiagent_particle_env.core import World, Agent, TurretAgent
from multiagent_particle_env.scenario import BaseScenario
from multiagent_particle_env.utils import pos_to_ang, pos_to_angle

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

        # Logging variables
        self.num_scores = None

    def make_world(self, args):
        """
        Construct the world

        Returns:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        self.args = args

        # Display debug output if already slowing simulation to display visual
        self.debug = self.debug or args.display

        # Create world and set properties
        world = World()
        world.turret = True
        world.dimension_communication = 2
        world.log_headers = ["Agent_Type", "Fixed", "Perturbed", "X", "Y", "dX", "dY", "fX", "fY", "entry_angle",
                             "theta", "theta_vel", "theta_x", "theta_y", "Score", "Collision"]
        num_landmarks = 0
        num_adversaries = args.num_adversaries
        num_good_agents = 1
        num_agents = num_adversaries + num_good_agents

        # All agents and the HVT have the same base size
        # size is in mm?
        factor = 0.25
        reduction_factor = 0.5
        size = (0.025 * factor * 1.5) * reduction_factor

        # Speed scalar for adversaries
        # A value of 1.0 cause the speed to be the same as turret
        # attacker_v/turret_v
        nu = self.args.nu

        # Multiplied by all speeds (only purpose is to slow down agents between timesteps)
        speed_factor = 1.0 / 4.0

        # Multiplied by max speed to get acceleration
        accel_factor = 10

        # Attacker
        world.agents = [Agent() for _ in range(num_adversaries)]
        for i in range(num_adversaries):
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
        turret.sense_region = (self.args.goal_radius + args.visibility_from_goal) * reduction_factor
        turret.size = .1 * reduction_factor
        turret.silent = True
        turret.head_color = np.array([0.25, 0.25, 0.25])
        turret.head_size = size
        if args.fixed_turret:
            turret.action_callback = fixed_turret
            # turret.is_fixed_policy = True
        world.agents.append(turret)

        self.num_scores = {agent: 0 for agent in world.agents}

        # Add boundary landmarks
        world.landmarks = world.landmarks + world.set_dense_boundaries()

        # Make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        adversaries = self.adversaries(world)

        # Set initial conditions for turret
        # Turret stays in center of environment
        world.agents[-1].state.p_pos = np.zeros(world.dimension_position)
        world.agents[-1].state.p_vel = np.zeros(world.dimension_position)
        world.agents[-1].state.p_placeholder_vel = np.zeros(world.dimension_position)
        world.agents[-1].state.c = np.zeros(world.dimension_communication)

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

    def reward(self, agent, world, dense=False, shared=True):
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
        if agent is world.agents[0]:
            self.calculate_win_lose_conditions(world)

        if agent.adversary:
            return self.adversary_reward(agent, world, dense, shared)
        else:
            return self.agent_reward(agent, world, dense)

    def agent_reward(self, agent, world, dense):
        """
        Turret reward

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            dense (boolean): Specifies whether to use dense reward

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
                reward += 20
                agent.score += 1
                adv.active = False
                adv.color = np.array([.545, .365, .176])
            elif adv.touched_turret:
                reward -= 10
                adv.active = False
                adv.color = np.array([0.35, 0.85, 0.35])

        return reward

    def adversary_reward(self, agent, world, dense, shared):
        """
        Recon reward

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            dense (boolean): Specifies whether to use dense reward
            shared (boolean): Specifies whether to use shared reward

        Returns:
            (float) Total agent reward
        """
        reward = 0
        adversaries = self.adversaries(world)
        turrets = self.good_agents(world)
        if not agent.active:
            if shared:
                # Agents position is irrelevant don't reward
                for adv in adversaries:
                    if not adv.active or adv is agent:
                        # Attacker cannot provide anymore reward
                        continue

                    # Calculate reward for adversary
                    # If the adversary has both hit and touched the turret
                    # the tie goes to the turret
                    if adv.hit:
                        reward -= 10
                    elif adv.touched_turret:
                        reward += 20
            return reward

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            reward -= world.bound(abs(agent.state.p_pos[coordinate_position]))

        # assert not (agent.touched_turret and agent.hit), "Adversary both touched turret and was hit."

        if dense:
            # Compute Dense Reward
            # Compute actual distance between entities
            delta_pos = agent.state.p_pos - turrets[0].state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_to_goal = (dist - agent.size) - turrets[0].goal_radius
            # print(f'{agent.name} - Dist to goal: {dist_to_goal}')

            # Save distance to goal region
            if (agent.prev_dist_to_goal is None) and (agent.new_dist_to_goal is None):
                agent.new_dist_to_goal = dist_to_goal

                if (agent.new_dist_to_goal < agent.inital_dist_to_goal):
                    reward += 0.1
            else:
                agent.prev_dist_to_goal = agent.new_dist_to_goal
                agent.new_dist_to_goal = dist_to_goal

                # Give dense reward if new dist to goal is less than previous dist to goal
                if (agent.new_dist_to_goal < agent.inital_dist_to_goal) and \
                        (agent.new_dist_to_goal < agent.prev_dist_to_goal):
                    reward += 0.1

        if shared:
            # Compute Shared Reward
            for adv in adversaries:
                if not adv.active:
                    # Attacker cannot achieve anymore reward
                    continue

                # Calculate reward for adversary
                # If the adversary has both hit and touched the turret
                # the tie goes to the turret
                if adv.hit:
                    reward -= 10
                elif adv.touched_turret:
                    reward += 20

            if agent.hit:
                agent.score -= 1
            elif agent.touched_turret:
                agent.score += 1

            # Update agent active state as needed
            # If the adversary has both hit and touched the turret
            # the tie goes to the turret
            # if agent.hit:
            #     agent.score -= 1
            #     agent.active = False
            #     agent.color = np.array([.545, .365, .176])
            # elif agent.touched_turret:
            #     agent.score += 1
            #     agent.active = False
            #     agent.color = np.array([0.35, 0.85, 0.35])

        else:
            # Compute individual reward
            # If the adversary has both hit and touched the turret
            # the tie goes to the turret
            if agent.hit:
                reward -= 10
                agent.score -= 1
                # agent.active = False
                # agent.color = np.array([.545, .365, .176])
            elif agent.touched_turret:
                reward += 20
                agent.score += 1
                # agent.active = False
                # agent.color = np.array([0.35, 0.85, 0.35])

        return reward

    def observation(self, agent, world):
        """
        Define the observations.
        all agents can only see active attackers

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (np.array) Observations array with the velocity of the agent, the position of the agent,
                       distance to all landmarks in the agent's reference frame,
                       distance to all other agents in the agent's reference frame,
                       and the velocities of the good agents.
        """
        adversaries = self.adversaries(world)
        other_pos = []
        other_vel = []

        if not agent.adversary:
            # Turret
            # [angular velocity, 0.0, turret head x, turret head y, turret body position x, turret body position y,
            # adv 0 x, adv 0 y, adv 1 x, adv 1 y, adv 0 vx, adv 0 vy, adv 1 vx, adv 1 vy]

            # allow for defender to see in turret sense region
            # if attacker goes into turret sense region then defender can see it for the rest of the episode
            for adversary in adversaries:
                if adversary.active and (not adversary.seen_by_turret) and world.in_sense_region(agent, adversary):
                    # First entry into turret sense region
                    adversary.seen_by_turret = True
                    adversary.initial_entry = True

                if adversary.active and adversary.seen_by_turret:
                    # Attacker position relative to HVT
                    other_pos.append(adversary.state.p_pos - agent.state.p_pos)

                    # Attacker velocity
                    other_vel.append(adversary.state.p_vel)

                else:
                    # Pad if attacker not seen or not active
                    other_pos.append(np.zeros(2))
                    other_vel.append(np.zeros(2))

            # 2 + 2 + 2 + 4 + 4 = 14
            return np.concatenate([np.array([agent.state.p_head_vel[0], 0.0])] + [agent.state.p_head_pos] +
                                  [agent.state.p_pos] + other_pos + other_vel).astype(np.float32)

        else:
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
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                                  turret_pos + turret_head_pos +
                                  other_pos + turret_head_vel + other_vel).astype(np.float32)

    def done(self, agent, world):
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

        if self.args.all_touch:
            # All attackers must have been hit or have touched the turret
            return all([not adversary.active for adversary in adversaries])
        else:
            # One attacker must have touched the turret or all attackers must have been hit
            return any([adversary.touched_turret and not adversary.hit for adversary in adversaries]) or \
                   all([adversary.hit for adversary in adversaries])

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
