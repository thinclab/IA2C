# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_hvt_1v1_random.py

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

from array import array
import numpy as np
import torch
import math
import pandas as pd
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
import random

# declare global variable
ATT_Sensing_region = 0.08
HVT_Size = 0.45
HVT_Sensing_region = 0.3
V = 0.7
Speed_Factor = 0.5
Map_size = 1
Agent_size = 0.02


class Scenario(BaseScenario):
    """
    Define the world, reward, and observations for the scenario.
    """

    def __init__(self):
        # Debug verbose output
        self.debug = False

    def make_world(self):  # , args): BiB: args not used
        """
        Construct the world

        Returns:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Debug verbose output
        self.debug = False

        # Boolean variable to detect if intruder enters HVT sensing range
        self.in_HVT = False
        # Create world and set properties
        world = World()
        world.dimension_communication = 2
        world.log_headers = ["Agent_Type", "Fixed", "Perturbed", "X", "Y", "dX", "dY", "fX", "fY", "Collision"]
        world.map_size = 1
        world.provide_global_obs = False
        # Defender and Attacker
        num_agents = 2

        # High Value Target (HVT)
        num_landmarks = 1

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]

        # All agents and the HVT have the same base size
        # size is in mm?
        factor = 0.25
        size = 0.025 * factor
        '''
        # Standard
        attacker_size = size
        attacker_sense_region_size = size * (20 / factor)

        defender_size = size
        defender_sense_region_size = size * (4 / factor)

        hvt_size = size + defender_sense_region_size
        hvt_sense_region_size = size * (20 / factor)

        # 50% of the HVT size
        defender_size = hvt_size * 0.5
        '''
        # ---------------------BiB------------------------
        attacker_size = size  # 1/160
        attacker_sense_region_size = ATT_Sensing_region  # size * (4 / factor) #1/10

        defender_size = size
        defender_sense_region_size = size * (4 / factor)  # 1/10

        world.def_sense_range = HVT_Sensing_region
        world.hvt_size = HVT_Size
        # 50% of the HVT size
        # defender_size = hvt_size * 0.5
        # ------------------------------------------------

        # Attacker
        world.agents[0].name = 'agent {}'.format(1)
        world.agents[0].adversary = True
        world.agents[0].accel = 3.0
        world.agents[0].collide = True
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        world.agents[0].has_sense = True
        world.agents[0].silent = True
        world.agents[0].sense_region = attacker_sense_region_size
        world.agents[0].size = Agent_size
        world.agents[0].max_speed = Speed_Factor * V
        world.adv_respawn_pos = 0.75

        # Defender
        # Sense region for defender is 20% smaller because
        # it has a speed advantage over the attacker
        world.agents[1].name = 'agent {}'.format(0)
        world.agents[1].accel = 4.0
        world.agents[1].collide = True
        world.agents[1].color = np.array([0.35, 0.85, 0.35])
        world.agents[1].has_sense = True
        world.agents[1].sense_region = 0.0
        world.agents[1].silent = True
        world.agents[1].sense_region = defender_sense_region_size
        world.agents[1].size = Agent_size
        world.agents[1].max_speed = Speed_Factor
        world.def_respawn_pos = 0.2
        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]

        # High Value Target (HVT)
        # Sense region for HVT is 20% larger because it cannot move
        world.landmarks[0].name = 'landmark {}'.format(0)
        world.landmarks[0].boundary = False
        world.landmarks[0].collide = False
        world.landmarks[0].color = np.array([0.25, 0.25, 0.25])
        world.landmarks[0].has_sense = True
        world.landmarks[0].movable = False
        world.landmarks[0].sense_region = world.hvt_size + world.def_sense_range#hvt_sense_region_size
        world.landmarks[0].size = world.hvt_size

        # Add boundary landmarks
        world.landmarks = world.landmarks + world.set_dense_boundaries()

        # Make initial conditions
        self.reset_world(world, 0)

        return world

    def engage_phase_pos(self):

        angle = random.uniform(0, 2 * math.pi)

        adv_pos_y = 1
        adv_pos_x = 1
        intruder_sensing_range = 0.1
        HVT_sensing_range = 0.5
        adv_pos_x = HVT_sensing_range * math.cos(angle)
        adv_pos_y = HVT_sensing_range * math.sin(angle)
        adv_pos = [adv_pos_x, adv_pos_y]
        angle = random.uniform(0, 2 * math.pi)
        def_pos_x = 1
        def_pos_y = 1
        while abs(def_pos_y) > HVT_sensing_range or abs(def_pos_x) > HVT_sensing_range:
            angle = random.uniform(0, 2 * math.pi)
            def_pos_x = adv_pos_x + intruder_sensing_range * math.cos(angle)
            def_pos_y = adv_pos_y + intruder_sensing_range * math.sin(angle)
        def_pos = [def_pos_x, def_pos_y]
        return adv_pos, def_pos

    def asym_phase_pos(self, Intruder_pos=None, Defender_pos=True):
        HVT_size = HVT_Size
        sensing_range = HVT_Sensing_region
        HVT_sensing_range = HVT_size + sensing_range
        intruder_sensing_region = ATT_Sensing_region
        intruder_x, intruder_y = 0, 0
        defender_x, defender_y = 0, 0
        if Intruder_pos == None:
            sample_dist = random.uniform(HVT_sensing_range**2, (HVT_sensing_range + intruder_sensing_region)**2)
            rho_intruder = math.sqrt(sample_dist)
            theta_intruder = random.uniform(0, 2*math.pi)
            intruder_x = rho_intruder * math.cos(theta_intruder)
            intruder_y = rho_intruder * math.sin(theta_intruder)
        else:
            intruder_x = Intruder_pos[0]
            intruder_y = Intruder_pos[1]

        while Defender_pos == True:
            # rho_defender = intruder_sensing_region * math.sqrt(random.random())
            theta_defender = random.uniform(0, 2*math.pi)
            defender_x = intruder_x + intruder_sensing_region * math.cos(theta_defender)
            defender_y = intruder_y + intruder_sensing_region * math.sin(theta_defender)

            if defender_x ** 2 + defender_y ** 2 <= HVT_sensing_range ** 2:
                break

        return [intruder_x, intruder_y], [defender_x, defender_y]

    def random_agents_inside_HVT(self, inside_HVT):
        '''
        0: defender randomly generated inside HVT
        1: defender randomly generated outside HVT but inside HVT sensing range
        2: defender randomly generated inside HVT sensing range
        '''
        HVT_size = HVT_Size
        sensing_range = HVT_Sensing_region
        HVT_sensing_range = HVT_size + sensing_range
        low_bound = -HVT_size  # HVT size
        high_bound = HVT_size
        rng = random.Random()
        # rng.seed(42)
        pos_x = 100
        if inside_HVT == 0:
            pos_y = rng.uniform(low_bound, high_bound)
            while pos_x * pos_x + pos_y * pos_y > pow(high_bound, 2):
                pos_x = rng.uniform(low_bound, high_bound)
            return [pos_x, pos_y]
        elif inside_HVT == 1:
            pos_y = rng.uniform(-1*HVT_sensing_range, HVT_sensing_range)
            while pos_x * pos_x + pos_y * pos_y > HVT_sensing_range * HVT_sensing_range or pos_x * pos_x + pos_y * pos_y < pow(high_bound, 2):
                pos_x = rng.uniform(-1*HVT_sensing_range, HVT_sensing_range)
            return [pos_x, pos_y]
        elif inside_HVT == 2:
            pos_y = rng.uniform(-1*HVT_sensing_range, HVT_sensing_range)
            while pos_x * pos_x + pos_y * pos_y > HVT_sensing_range * HVT_sensing_range:
                pos_x = rng.uniform(-1*HVT_sensing_range, HVT_sensing_range)
            return [pos_x, pos_y]

    def random_agents_with_constrain_intruer(self, world):
        HVT_size = HVT_Size
        sensing_range = HVT_Sensing_region
        HVT_sensing_range = HVT_size + sensing_range + 0.1
        rng = random.Random()

        def inside_HVT():
            theta = np.random.uniform(0, 2*np.pi)

            # r = np.sqrt(np.random.uniform(self.adv_respawn_pos**2, (HVT_size + sensing_range)**2))
            r = world.adv_respawn_pos
            pos_x = r * np.cos(theta)
            pos_y = r * np.sin(theta)
            return [pos_x, pos_y]

        def outside_HVT():
            # intruder initial position
            low_bound = -Map_size  # map max size
            high_bound = Map_size
            pos_x, pos_y = 0, 0
            dist_sq = (pos_x)**2 + (pos_y)**2
            while dist_sq < HVT_sensing_range**2:
                pos_x = np.random.uniform(low_bound, high_bound)
                pos_y = np.random.uniform(low_bound, high_bound)
                dist_sq = (pos_x)**2 + (pos_y)**2

            return [pos_x, pos_y]
        # if world.adv_respawn_pos < HVT_sensing_range:
        #     return inside_HVT()
        # else:
        return outside_HVT()

    def sample_positions_with_sensing(self, world, hvt_pos, intruder_r, defender_r, seed=None):
        """
        Sample (intruder, defender) positions based on three parameters:
          - If intruder_r ≤ sensing_range: intruder is on the circle of radius intruder_r centered at HVT.
            Otherwise: intruder is sampled uniformly in area from the annulus [sensing_range, intruder_r].
          - If defender_r ≤ sensing_range: defender is on the circle of radius defender_r centered at intruder.
            Otherwise: defender is sampled uniformly in area inside the sensing_range disk centered at HVT.
        """

        def _sample_on_circle(center, radius, rng):
            theta = rng.uniform(0.0, 2 * np.pi)
            return np.array([
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta)
            ], dtype=float)

        def _sample_in_annulus(center, r_min, r_max, rng):
            # Area-uniform sampling in an annulus: sample in [r_min^2, r_max^2] uniformly, then take sqrt
            r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2))
            theta = rng.uniform(0.0, 2 * np.pi)
            return np.array([
                center[0] + r * np.cos(theta),
                center[1] + r * np.sin(theta)
            ], dtype=float)

        def _sample_in_disk(center, r_max, rng):
            # Area-uniform sampling in a disk: sample radius^2 uniformly, then take sqrt
            r = np.sqrt(rng.uniform(0.0, r_max ** 2))
            theta = rng.uniform(0.0, 2 * np.pi)
            return np.array([
                center[0] + r * np.cos(theta),
                center[1] + r * np.sin(theta)
            ], dtype=float)

        def _sample_on_circle_inside_disk(intr, hvt, defender_r, sensing_range, rng):
            """
            Sample a point on the circle centered at intr with radius defender_r, such that
            the point lies inside the disk centered at hvt with radius sensing_range.
            If there is no feasible solution, return a fallback.
            """
            dx, dy = hvt[0] - intr[0], hvt[1] - intr[1]
            d = np.hypot(dx, dy)

            # Special case: intruder coincides with HVT
            if d == 0.0:
                if defender_r <= sensing_range:
                    # The entire defender circle is within the HVT sensing disk
                    return _sample_on_circle(intr, defender_r, rng)
                else:
                    # No feasible circle point; sample anywhere in the HVT sensing disk instead
                    return _sample_in_disk(hvt_pos, sensing_range, rng)

            # From the cosine rule: cos(alpha) >= k
            # alpha is the angle between defender direction and (intr -> HVT) direction
            k = (defender_r ** 2 + d ** 2 - sensing_range ** 2) / (2.0 * defender_r * d)

            if k > 1.0:
                # No angle satisfies the condition (circle is completely outside the disk)
                return np.array([0.0, 0.0])  # fallback: invalid marker or adjust as needed
            elif k <= -1.0:
                # Entire circle lies inside the disk: any angle is valid
                return _sample_on_circle(intr, defender_r, rng)
            else:
                # Only a circular arc around the direction intr->HVT is feasible
                arc = np.arccos(k)  # half-width of the feasible angle interval
                base = np.arctan2(dy, dx)  # direction from intr to HVT
                theta = base + rng.uniform(-arc, arc)  # sample uniformly within feasible arc
                return np.array([intr[0] + defender_r * np.cos(theta),
                                 intr[1] + defender_r * np.sin(theta)], dtype=float)

        rng = np.random.default_rng(seed)
        sensing_range = HVT_Size + HVT_Sensing_region
        # --- intruder ---
        # if intruder_r <= sensing_range:
        #     intr = _sample_on_circle(hvt_pos, intruder_r, rng)
        # else:
        #     intr = _sample_in_annulus(hvt_pos, sensing_range, 1.0, rng)
        intr = self.sample_intruder_pos_in_quadrants(world, allowed_quadrants=(1, 2, 3, 4))
        # --- defender ---
        defdr = _sample_on_circle_inside_disk(intr, [0.0, 0.0], defender_r, sensing_range, rng)

        return intr, defdr

    def sample_intruder_pos_in_quadrants(self, world, allowed_quadrants=(1, 2)):
        """
        Sample intruder spawn position only inside specified quadrants.

        Quadrant index:
            1: x > 0, y > 0
            2: x < 0, y > 0
            3: x < 0, y < 0
            4: x > 0, y < 0

        world.adv_respawn_pos: inner radius or range for intruder spawn (you already use this)
        world.map_size: outer map radius or bound (if exists)

        allowed_quadrants: tuple/list of quadrant indices to sample from.
        """
        # choose which quadrant this spawn will use
        q = int(np.random.choice(allowed_quadrants))

        # decide sign of x,y based on quadrant
        if q == 1:
            sx, sy = +1.0, +1.0
        elif q == 2:
            sx, sy = -1.0, +1.0
        elif q == 3:
            sx, sy = -1.0, -1.0
        elif q == 4:
            sx, sy = +1.0, -1.0
        else:
            # fallback: treat as full circle if misconfigured
            sx, sy = 1.0, 1.0

        # you probably already have something like adv_respawn_pos as a radius
        # here we treat it as "min radius", and use map_size as "max radius" if available
        r_min = getattr(world, "adv_respawn_pos", 0.5)
        r_max = getattr(world, "map_size", 1.0)

        # sample radius in [r_min, r_max]
        r = np.random.uniform(r_min, r_max)

        # sample x,y in that quadrant with approximate radius r
        # Option 1: polar-based sampling restricted to quadrant
        #   angle_range for quadrant:
        #     Q1: [0, π/2), Q2: [π/2, π), Q3: [π, 3π/2), Q4: [3π/2, 2π)
        if q == 1:
            theta = np.random.uniform(0.0, 0.5 * np.pi)
        elif q == 2:
            theta = np.random.uniform(0.5 * np.pi, np.pi)
        elif q == 3:
            theta = np.random.uniform(np.pi, 1.5 * np.pi)
        else:  # q == 4
            theta = np.random.uniform(1.5 * np.pi, 2.0 * np.pi)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # if your map is a square and you just want "inside map", you can also
        # clamp/scale here if needed.
        return np.array([x, y], dtype=np.float32)


    def sample_intruder_pos_q1_outside_sense(self, world,
                                             r_margin: float = 1e-3,
                                             r_max: float = 0.95,
                                             adv_respawn_pos = None):
        """
        Sample an intruder spawn position outside the HVT sensing range and in the
        first quadrant (relative to the HVT).
        world: env.world
        r_margin: small margin added beyond the sensing region to avoid numerical edge cases
        r_max: maximum respawn radius (can be adjusted based on map size)
        """
        # hvt = world.landmarks[0]
        # cx, cy = hvt.state.p_pos  # HVT position
        # r_min = float(hvt.sense_region) + r_margin

        # # If the world defines a custom respawn radius, use it as r_max
        # # if hasattr(world, "adv_respawn_pos"):
        # #     r_max = max(r_min + 1e-3, float(world.adv_respawn_pos))

        # # Simple rejection sampling: try several times to avoid rare out-of-bounds cases
        # for _ in range(128):
        #     # Radius sampled uniformly from [r_min, r_max]
        #     if adv_respawn_pos == None:
        #         r = np.random.uniform(r_min, r_max)
        #     else:
        #         r = adv_respawn_pos
        #     # Angle sampled from [0, π/2], i.e., first quadrant
        #     theta = np.random.uniform(0.0, 0.5 * np.pi)
        #     x = cx + r * np.cos(theta)
        #     y = cy + r * np.sin(theta)

        #     # If the map is [-0.5, 0.5]×[-0.5, 0.5], enforce these bounds here
        #     # Assuming a 1x1 map with center at (0, 0):
        #     if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
        #         print('COOL')
        #         return np.array([x, y], dtype=np.float32)

        # # Fallback: if no valid point is found, place intruder just outside sensing region along the diagonal
        # return np.array([cx + r_min, cy + r_min], dtype=np.float32)
        theta = np.random.uniform(0.0, np.pi / 2.0)  # [0, π/2]
        x = adv_respawn_pos * np.cos(theta)
        y = adv_respawn_pos * np.sin(theta)
        return np.array([x, y], dtype=np.float32)

    def reset_world(self, world, restart_in_HVT):
        """
        Reset the world to the initial conditions.

        Args:
            world (multiagent_particle_env.core.World): World object with agents and landmarks
        """
        # Set random initial states for HVT
        for landmark in world.landmarks:
            if not landmark.boundary:
                # landmark.state.p_pos = np.random.uniform(-0.39, +0.39, world.dimension_position)
                landmark.state.p_pos = np.zeros(world.dimension_position)  # BiB: changed to fix at origin
                landmark.state.p_vel = np.zeros(world.dimension_position)

        # TODO: Set pseudo-random positions for defender and attacker dependent on HVT

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dimension_position)
            agent.state.p_vel = np.zeros(world.dimension_position)
            agent.state.c = np.zeros(world.dimension_communication)

        # Set HVT sensing range
        world.landmarks[0].size = world.hvt_size
        world.landmarks[0].sense_region = world.hvt_size + world.def_sense_range
        # BiB: set fixed initial locations
        self.in_HVT = False
        if not restart_in_HVT:
            att_pos, def_pos = self.sample_positions_with_sensing(world,[0.0, 0.0], world.adv_respawn_pos,
                                                                  world.def_respawn_pos)
            world.agents[1].state.p_pos = np.asarray(def_pos)#self.random_agents_inside_HVT(0)
            world.agents[0].state.p_pos = np.asarray(att_pos)
        else:
            att_pos, def_pos = self.sample_positions_with_sensing(world,[0.0, 0.0], world.adv_respawn_pos,
                                                                  world.def_respawn_pos)
            world.agents[1].state.p_pos = np.asarray([0.0, 0.0])
            world.agents[0].state.p_pos = np.asarray(att_pos)#np.asarray(self.random_agents_with_constrain_intruer(world))

    def good_agents(self, world):
        """
        Returns all agents that are not adversaries in a list.

        Returns:
            (list) All the agents in the world that are not adversaries.
        """
        return [agent for agent in world.agents if not agent.adversary]

    def angle_asym(self, center, intruder, defender):
        xc, yc = center
        x1, y1 = intruder
        x2, y2 = defender

        # vectors from center
        v1 = (x1 - xc, y1 - yc)
        v2 = (x2 - xc, y2 - yc)

        # dot product and norms
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm = math.hypot(*v1) * math.hypot(*v2)

        # guard against divide-by-zero or tiny rounding errors
        cos_theta = max(-1.0, min(1.0, dot / norm))
        theta = math.acos(cos_theta)
        return theta

    def angle_xaxis(self, intruder):
        pos = intruder.state.p_pos
        angle = math.atan2(pos[1], pos[0])
        return angle * 180 / math.pi

    def enter_HVT(self, world):
        hvt = [landmark for landmark in world.landmarks if not landmark.boundary][0]
        intruder = self.adversaries(world)[0]
        defender = self.good_agents(world)[0]
        angle = None
        if not self.in_HVT:
            if world.in_sense_region(hvt, intruder):
                # angle = self.angle_asym([0, 0], intruder.state.p_pos, defender.state.p_pos)
                angle = self.angle_xaxis(intruder)
                self.in_HVT = True
        else:
            if not world.in_sense_region(hvt, intruder):
                self.in_HVT = False
        return angle

    def adversaries(self, world):
        """
        Returns all agents that are adversaries in a list.

        Returns:
            (list) All the agents in the world that are adversaries.
        """
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world, dense=True):
        """
        Reward is based on prey agent not being caught by predator agents.

        Good agents are negatively rewarded if caught by adversaries and for exiting the screen.

        Adversaries are rewarded for collisions with good agents.



        Dense reward at start (get other agent in sense region)

        and

        Sparse reward after


        Give small reward for getting other agent in your sense region
        Give larger reward for completion of objective




        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            dense (boolean): Specifies whether to use dense reward

        Returns:
            If agent is adversary:
                self.adversary_reward() result
            Else:
                self.agent_reward() result
        """
        if agent.adversary:
            return self.adversary_reward(agent, world, dense)
        else:
            return self.agent_reward(agent, world, dense)

    def agent_reward(self, agent, world, dense):
        """
        Defender reward

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            dense (boolean): Specifies whether to use dense reward

        Returns:
            (float) Total agent reward
        """

        intruder = self.adversaries(world)[0]
        landmarks = [landmark for landmark in world.landmarks if not landmark.boundary]
        step_r, shape_r, final_r = 0, 0, 0

        # Determine collisions with attackers, assign reward
        source = torch.tensor([
            agent,
            *([intruder]*len(landmarks))
        ])
        targets = [
            intruder.state.p_pos,
            *[l.state.p_pos for l in landmarks]
        ]
        collisions = world.is_collision(source, targets)

        if collisions[1:].any():
            final_r = -10

        if collisions[0]:
            final_r = 10
        else:
            step_r = -0.01


        # Determine Attacker collision with HVT and assign penalty

        for hvt in landmarks:
            if world.is_collision(intruder, hvt):
                final_r = -10  # - ((1 - cur_dist)/1 * 2)

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            shape_r -= world.bound(abs(agent.state.p_pos[coordinate_position]))

        return [step_r, shape_r, final_r]

    def adversary_reward(self, agent, world, dense):
        """
        Attacker reward

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks
            dense (boolean): Specifies whether to use dense reward

        Returns:
            (float) Total agent reward
        """
        defender = self.good_agents(world)[0]
        reward = 0
        agents = self.good_agents(world)
        landmarks = [landmark for landmark in world.landmarks if not landmark.boundary]
        step_r, shape_r, final_r = 0, 0, 0

        # Determine collisions with defenders, assign penalties

        if world.is_collision(agent, defender):
            final_r = -10

        # Determine Attacker collision with HVT and assign reward
        for hvt in landmarks:
            if world.is_collision(agent, hvt):
                final_r = 10
            else:
                step_r = -0.01

        # Determine if agent left the screen and assign penalties
        for coordinate_position in range(world.dimension_position):
            shape_r -= world.bound(abs(agent.state.p_pos[coordinate_position]))

        enter_angle = self.enter_HVT(world)

        return [step_r, shape_r, final_r, enter_angle]

    def observation(self, agent, world):
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
        def nearest_point_on_circle(point, circle, r):
            px, py = point[0], point[1]
            cx, cy = circle[0], circle[1]
            vx, vy = px - cx, py - cy
            d = math.hypot(vx, vy)  # distance between point to circle center
            scale = r / d
            return [cx + vx * scale, cy + vy * scale]

        intruder = self.adversaries(world)[0]
        defender = self.good_agents(world)[0]
        hvt = None
        for landmark in world.landmarks:
            if not landmark.boundary:
                hvt = landmark
                break

        # Get positions of HVT in this agent's reference frame
        landmarks_pos = [array('d', [0, 0])]
        hvt_sense_pos = [array('d', [0, 0])]
        other_pos = [array('d', [0, 0])]
        other_vel = [array('d', [0, 0])]
        self_pos = [array('d', [0, 0])]
        self_vel = [array('d', [0, 0])]
        sensed_by_HVT = [array('d', [0])]
        # obs for intruder
        # print(landmark.state.p_pos - intruder.state.p_pos)
        if agent.adversary:
            if world.in_sense_region(intruder, hvt):
                # nearest_point = nearest_point_on_circle(intruder.state.p_pos, hvt.state.p_pos,
                #                                         hvt.sense_region)
                sensed_by_HVT = [array('d', [1])]
            landmarks_pos = [hvt.state.p_pos - intruder.state.p_pos]
            if world.in_sense_region(intruder, defender):
                other_pos = [defender.state.p_pos - intruder.state.p_pos]
                #other_vel = [defender.state.p_vel]
            self_pos = [intruder.state.p_pos]
            self_vel = [intruder.state.p_vel]
            return np.concatenate(self_pos + self_vel + landmarks_pos + other_pos + sensed_by_HVT)

        else:
            if world.in_sense_region(hvt, intruder):
                hvt_sense_pos = [intruder.state.p_pos - hvt.state.p_pos]
                landmarks_pos = [hvt.state.p_pos - defender.state.p_pos]
                other_pos = [intruder.state.p_pos - defender.state.p_pos]
                #other_vel = [intruder.state.p_vel]
            self_pos = [defender.state.p_pos]
            self_vel = [defender.state.p_vel]
            # print(self_vel , self_pos , landmarks_pos , other_pos , other_vel , hvt_sense_pos)
            return np.concatenate(self_pos + self_vel + landmarks_pos + other_pos + hvt_sense_pos)

    def done(self, agent, world):
        """
        Determines whether the terminal condition for the episode has been reached.

        Args:
            agent (multiagent_particle_env.core.Agent): Agent object
            world (multiagent_particle_env.core.World): World object with agents and landmarks

        Returns:
            (bool) Terminal condition reached flag
        """
        adversaries = self.adversaries(world)
        landmarks = [landmark for landmark in world.landmarks if not landmark.boundary]
        attacker_flag = False
        defender_flag = False

        # Determine collisions with defenders, assign penalties
        if agent.adversary:
            for hvt in landmarks:
                if world.is_collision(agent, hvt):
                    attacker_flag = True
        else:
            for adv in adversaries:
                if world.is_collision(agent, adv):
                    defender_flag = True

        return attacker_flag or defender_flag

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

        # Check for collisions
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if agent in good_agents:
            agent_type = "Defender"
            for adv in adversaries:
                if world.is_collision(agent, adv):
                    collision += 1
        elif agent in adversaries:
            agent_type = "Attacker"
            for ga in good_agents:
                if world.is_collision(agent, ga):
                    collision += 1
        else:
            collision = "N/A"

        log_data = [agent_type, fixed, perturbed, x, y, dx, dy, fx, fy, collision]

        return log_data
