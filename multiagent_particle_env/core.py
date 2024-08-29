# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
core.py

Contains the core classes of the Multi-Agent Particle Environment

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

import numpy as np

from multiagent_particle_env.utils import pos_to_ang, pos_to_angle

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


class EntityState(object):
    """
    Physical/External Base State of All Entities
    """

    def __init__(self):
        # Grid Position if using discrete grid [np.array/list]
        self.g_pos = None

        # Physical Position (p_pos) [np.array]
        self.p_pos = None

        # Physical Velocity (p_vel) [np.array]
        self.p_vel = None


class AgentState(EntityState):
    """
    State of Agents (Including Communication and Internal/Mental state)
    """

    def __init__(self):
        super(AgentState, self).__init__()
        # Communication Utterance (c) [np.array]
        self.c = None

        # State list
        self.state = None


class TurretAgentState(AgentState):
    """
    State of Agents (Including Communication and Internal/Mental state)
    """

    def __init__(self):
        super(TurretAgentState, self).__init__()
        # Physical Angular Position (p_theta) [np.array]
        self.p_head_theta = None

        # Physical Tracking Position (p_pos) [np.array]
        self.p_head_pos = None

        # Physical Tracking Velocity (p_vel) [np.array]
        self.p_head_vel = None

        # Physical Placeholder Position for turret head calculation (p_pos) [np.array]
        self.p_placeholder_pos = None

        # Physical Placeholder Velocity for turret head calculation (p_vel) [np.array]
        self.p_placeholder_vel = None


class Action(object):
    """
    Action of the Agent
    """

    def __init__(self):
        # Communication Action (c)
        self.c = None

        # Physical Action (u)
        self.u = None

        # Optional capture action when discrete actions are used
        self.capture = None

        # Raw discrete action from network
        self.discrete_raw = None


class Entity(object):
    """
    Properties and State of Physical World Entity
    """

    def __init__(self):
        # Acceleration
        self.accel = None

        # Entity Collides with Others
        self.collide = True

        # Color
        self.color = None

        # Material Density (Affects Mass)
        self.density = 25.0

        # Entity has sensing capability
        self.has_sense = False

        # Index
        self.index = None

        # Mass
        self.initial_mass = 1.0

        # Agent or not
        self.is_agent = False

        # Max Speed
        self.max_speed = None

        # Entity can Move or be Pushed
        self.movable = False

        # Name
        self.name = ''

        # Sensing region size
        self.sense_region = .200

        # Size
        self.size = 0.050

        # State
        self.state = EntityState()

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):
    """
    Properties of Landmark Entities
    """

    def __init__(self):
        super(Landmark, self).__init__()
        # Boundary marker
        self.boundary = False


class Agent(Entity):
    """
    Properties of agent Entities
    """

    def __init__(self):
        super(Agent, self).__init__()
        # Action
        self.action = Action()

        # Script behavior to execute
        self.action_callback = None

        # Adversarial agent
        self.adversary = False

        # Cannot observe the world
        self.blind = False

        # Communication (c) noise amount
        self.c_noise = None

        # Counter
        self.counter = None

        # Goal landmark
        self.goal_a = None
        self.goal_b = None

        # Identify as agent
        self.is_agent = True

        # Agent uses fixed policy
        self.is_fixed_policy = False

        # Agent uses altered previous policy
        self.is_perturbed_policy = False

        # Leader agent
        self.leader = False

        # Agents are movable by default
        self.movable = True

        # Agent reward
        self.reward = 0

        # Sensing region size
        self.sense_region = .100

        # Cannot send communication signals
        self.silent = False

        # Sends messages
        self.speaker = False

        # State
        self.state = AgentState()

        # Physical (u) motor noise amount
        self.u_noise = None

        # Physical (u) Control range
        self.u_range = 1.0

        # Turret scenario
        self.inital_dist_to_goal = None
        self.prev_dist_to_goal = None
        self.new_dist_to_goal = None
        self.prev_dist = None
        self.new_dist = None
        self.seen_by_turret = False
        self.initial_entry = False
        self.hit = False
        self.touched_turret = False
        self.active = True
        self.score = 0


class TurretAgent(Agent):
    """
    Properties of agent Entities
    """

    def __init__(self):
        super(TurretAgent, self).__init__()

        # Entity Collides with Others
        self.collide = False

        # Turrets Agents are not movable
        # they rotate instead
        self.movable = False

        # Turret agent rotates
        # Movable must be false
        self.rotating = True

        # State
        self.state = TurretAgentState()

        # Goal radius
        self.goal_radius = 0.25

        # Tracking Color
        self.head_color = None

        # Tracking Size
        self.head_size = 0.050

        assert self.collide is False, 'Turret collision not implemented'

        if self.rotating:
            assert self.movable is False, 'If agent rotates is cannot be movable'


class World(object):
    """
    Multi-Agent World
    """
    def __init__(self):
        # List of Agents and Entities (Can change at execution-time!)
        self.agents = []
        self.food = []
        self.forests = []
        self.landmarks = []
        self.random_agents = []
        self.stationary_agents = []

        # Pass along the scenario
        self.scenario = None

        # Contact response parameters
        self.apply_contact_forces = True
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # Whether agents share rewards
        self.collaborative = False

        # Physical damping
        self.damping = 0.25

        # Color dimensionality
        self.dimension_color = 3

        # Communication Channel Dimensionality
        self.dimension_communication = 0

        # Position dimensionality
        self.dimension_position = 2

        # Simulation timestep
        self.dt = 0.1

        # Logging headers
        self.log_headers = []

        # Turret Scenario
        self.total_scores = 0

    @property
    def entities(self):
        """
        Returns all entities in the world in a single concatenated list.

        Returns:
            (list) All the entities in the world
        """
        return self.agents + self.landmarks + self.stationary_agents + self.random_agents

    @property
    def active_agents(self):
        """
        Returns all active agents controllable by external policies in a list.

        Returns:
            (list) All the active agents in the world controllable by external policies.
        """
        return [agent.name for agent in self.agents if agent.active]

    @property
    def policy_agents(self):
        """
        Returns all agents controllable by external policies in a list.

        The additional conditional checks for agent.is_perturbed_policy and
        agent.is_fixed allow for the original model policies to be loaded by
        tensorflow to maintain the proper tensor checks. The actions of the
        policy for fixed or perturbed agents are ultimately ignored and
        instead the fixed or perturbed policy based actions are used instead.
        As they agents are also considered scripted agents and have their
        policy actions overwritten when the world is advanced a step.

        Returns:
            (list) All the agents in the world controllable by external policies.
        """
        return [agent for agent in self.agents if agent.action_callback is None or
                agent.is_perturbed_policy or agent.is_fixed_policy]

    @property
    def scripted_agents(self):
        """
        Returns all agents controlled by world scripts in a list.

        Returns:
            (list) All the agents in the world controllable by world scripts.
        """
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        """
        Update the world state
        """
        # Set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # Gather forces applied to entities
        p_force = [None] * len(self.entities)

        stag_hunt_scenario = self.stag_hunt if hasattr(self, 'stag_hunt') else False
        if not stag_hunt_scenario:
            # Apply agent physical controls
            p_force = self.apply_action_force(p_force)

            # Apply environment forces
            if self.apply_contact_forces:
                p_force = self.apply_environment_force(p_force)

            # Integrate physical state
            self.integrate_state(p_force)
        else:
            # Apply and integrate physical action
            self.scenario.apply_and_integrate_action(self)

        # Update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    def apply_action_force(self, p_force):
        """
        Gathers Agent action forces for movable agents based on the actions taken by the agents
        with noise applied when specified in the agents properties.

        Args:
            p_force (list): Physical forces applied to entities. All entries all 'None' initially.
                            Length of list is equal to the number of entities in the world.

        Returns:
            (list) Physical forces applied to entities with the forces for movable agents
                   updated based on their action with noise applied when specified.
                   Length of list is equal to the number of entities in the world.
        """
        # Set applied forces for movable agents
        for i, agent in enumerate(self.agents):
            if agent.movable or isinstance(agent, TurretAgent):
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0

                # Check if agent action is None
                if agent.action is None:
                    print("apply_action_force: ERROR, Agent action is none")

                p_force[i] = agent.action.u + noise

                # Debug Statement
                # print("Agent {}: {}".format(i, agent.action.u))
        return p_force

    def apply_environment_force(self, p_force):
        """
        Gather physical forces acting on entities based on the interactions between the different
        entities. Currently, only collisions forces are considered.

        Args:
            p_force (list): Physical forces applied to entities. All entries all 'None' initially except those
                            of movable agents which were updated by the function call to self.apply_action_force().
                            Length of list is equal to the number of entities in the world.

        Returns:
            (list) Physical forces applied to entities with the forces for all entities
                   updated based on their interactions with other entities in the world.
                   Length of list is equal to the number of entities in the world.
        """
        # TODO: Implement more efficient collision response
        # Simple (but inefficient) Collision Response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                # (1) Entities cannot collide with themselves and
                # (2) If entity_a is less than entity_b, it has already been previously evaluated.
                if b <= a:
                    continue

                # Calculate collision force between entity_a and entity_b
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)

                # Update physical forces for entity_a
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0

                    p_force[a] = f_a + p_force[a]

                # Update physical forces for entity_b
                if f_b is not None:
                    if p_force[b] is None: p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    def integrate_state(self, p_force):
        """
        Integrate physical state of all entities.

        Args:
            p_force (list): Physical forces applied to entities. All entries have been updated
                            by the function calls to self.apply_action_force() and
                            self.apply_environment_force() [If apply contact forces is enabled for the world].
                            Length of list is equal to the number of entities in the world.
        """
        for i, entity in enumerate(self.entities):
            if isinstance(entity, TurretAgent):
                # Physical forces don't apply to perturbed policy agents
                if entity.is_agent and entity.is_perturbed_policy:
                    continue

                # Scenario 1: Consider network output in polar space
                # entity.state.p_head_vel = entity.state.p_head_vel * (1 - self.damping)
                # if p_force[i] is not None:
                #     entity.state.p_head_vel += (p_force[i][0] / entity.mass) * self.dt
                #
                # if entity.max_speed is not None:
                #     if entity.state.p_head_vel > entity.max_speed:
                #         entity.state.p_head_vel = (entity.state.p_head_vel /
                #                                    (entity.state.p_head_vel * entity.max_speed))
                #
                # entity.state.p_head_theta += entity.state.p_head_vel * self.dt
                #
                # # Ensure value of entity.state.p_head_theta stays within 0 to 2*PI
                # p_head_theta = np.degrees(entity.state.p_head_theta)
                # if p_head_theta > 360.0:
                #     while p_head_theta > 360.0:
                #         p_head_theta -= 360.0
                # elif p_head_theta < 0.0:
                #     while p_head_theta < 0.0:
                #         p_head_theta = 360.0 + entity.state.p_head_theta
                # entity.state.p_head_theta = np.radians(p_head_theta)
                #
                # entity.state.p_head_pos = np.concatenate([np.cos(entity.state.p_head_theta) * entity.size,
                #                                           np.sin(entity.state.p_head_theta) * entity.size])

                # Scenario 2: Consider network output in cartesian space
                entity.state.p_placeholder_vel = entity.state.p_placeholder_vel * (1 - self.damping)
                if p_force[i] is not None:
                    entity.state.p_placeholder_vel += (p_force[i] / entity.mass) * self.dt

                if entity.max_speed is not None:
                    speed = np.sqrt(np.square(entity.state.p_placeholder_vel[0]) +
                                    np.square(entity.state.p_placeholder_vel[1]))

                    if speed > entity.max_speed:
                        entity.state.p_placeholder_vel = (entity.state.p_placeholder_vel /
                                                          np.sqrt(np.square(entity.state.p_placeholder_vel[0]) +
                                                                  np.square(entity.state.p_placeholder_vel[1])) *
                                                          entity.max_speed)

                entity.state.p_placeholder_pos += entity.state.p_placeholder_vel * self.dt

                # Scenario 1: x,y to Theta with arctan logic
                # entity.state.p_head_vel = np.array([entity.state.p_placeholder_vel[0],
                #                                     entity.state.p_placeholder_vel[1]])
                # entity.state.p_head_theta = np.array([pos_to_ang(entity.state.p_placeholder_pos)])

                # Scenario 2: x to Theta with module 2PI logic
                entity.state.p_head_vel = np.array([entity.state.p_placeholder_vel[0]])
                entity.state.p_head_theta = np.array([entity.state.p_placeholder_pos[0] % (2 * np.pi)])

                entity.state.p_head_pos = np.concatenate([np.cos(entity.state.p_head_theta) * entity.size,
                                                          np.sin(entity.state.p_head_theta) * entity.size])
            else:
                # Physical forces don't affect non-movable entities
                if not entity.movable:
                    continue

                # Physical forces don't apply to perturbed policy agents
                if entity.is_agent and entity.is_perturbed_policy:
                    continue

                entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
                if p_force[i] is not None:
                    entity.state.p_vel += (p_force[i] / entity.mass) * self.dt

                if entity.max_speed is not None:
                    speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))

                    if speed > entity.max_speed:
                        entity.state.p_vel = (entity.state.p_vel /
                                              np.sqrt(np.square(entity.state.p_vel[0]) +
                                                      np.square(entity.state.p_vel[1])) * entity.max_speed)

                entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        """
        Update the state of the agent

        Args:
            agent (multiagent_particle_env.Agent): An agent object.
        """
        # Set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dimension_communication)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    def bound(self, coordinate_position):
        """
        Penalty for agents exiting the screen, so that they can be caught by the adversaries.

        Args:
            coordinate_position (float): X or Y coordinate position

        Returns:
            (int or float) Penalty for agent exiting the screen
        """
        if coordinate_position < 0.9:
            return 0

        if coordinate_position < 1.0:
            return (coordinate_position - 0.9) * 10

        return min(np.exp(2 * coordinate_position - 2), 10)

    def set_boundaries(self):
        """
        Creates a boundary edge using multiagent_particle_env.core.Landmark objects.

        Returns:
            (list) A list of multiagent_particle_env.core.Landmark objects that make up the boundary edges of the world.
        """
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)

        # Landmarks for x-Coordinate Plane
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                boundary = Landmark()
                boundary.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(boundary)

        # Landmarks for y-Coordinate Plane
        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                boundary = Landmark()
                boundary.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(boundary)

        # Set properties for boundary landmarks
        for i, boundary in enumerate(boundary_list):
            boundary.name = 'boundary {}'.format(i)
            boundary.collide = True
            boundary.movable = False
            boundary.boundary = True
            boundary.color = np.array([0.75, 0.75, 0.75])
            boundary.size = landmark_size
            boundary.state.p_vel = np.zeros(self.dimension_position)

        return boundary_list

    def set_dense_boundaries(self):
        """
        Creates a boundary edge using multiagent_particle_env.core.Landmark objects.

        Returns:
            (list) A list of multiagent_particle_env.core.Landmark objects that make up the boundary edges of the world.
        """
        # Create boundary landmarks
        boundaries = [Landmark() for i in range(42)]
        for i, bound in enumerate(boundaries):
            bound.name = 'boundary {}'.format(i)
            bound.boundary = True
            bound.collide = True
            bound.color = np.array([0.25, 0.25, 0.25])
            bound.movable = False
            bound.size = 3.0

        # Set random initial states for landmarks and boundary landmarks
        # x = np.arange(-1.0,1.1,0.2).tolist()
        # y =  np.arange(-1.0,1.1,0.2).tolist()
        x = np.linspace(-1.0, 1.1, 11).tolist()
        y = np.linspace(-1.0, 1.1, 11).tolist()
        boundary_pos1 = np.array(np.meshgrid(x, np.array([-1.0, 1.0]))).T.reshape(-1, 2).tolist()
        boundary_pos2 = np.array(np.meshgrid(np.array([-1.0, 1.0]), y)).T.reshape(-1, 2).tolist()
        boundary_pos = np.array(boundary_pos1 + boundary_pos2)

        for landmark in boundaries:
            landmark.state.p_pos, boundary_pos = boundary_pos[-1], boundary_pos[:-1]
            if abs(landmark.state.p_pos[0]) >= 1.0:
                landmark.state.p_pos[0] = landmark.state.p_pos[0] + np.sign(landmark.state.p_pos[0]) * landmark.size
            if abs(landmark.state.p_pos[1]) >= 1.0:
                landmark.state.p_pos[1] = landmark.state.p_pos[1] + np.sign(landmark.state.p_pos[1]) * landmark.size

        return boundaries

    def is_collision(self, agent_a, agent_b):
        """
        Determine whether two agents collided.

        Args:
            agent_a (multiagent_particle_env.core.Agent): Agent object
            agent_b (multiagent_particle_env.core.Agent): Agent object

        Returns:
            (boolean) True if collision occurred else False
        """
        # Compute actual distance between entities
        delta_pos = agent_a.state.p_pos - agent_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        # Minimum allowable distance
        dist_min = agent_a.size + agent_b.size

        # Collision occurs is distance is less then the minimum allowable distance
        return True if dist < dist_min else False

    def in_sense_region(self, agent_a, agent_b):
        """
        Determine whether agent b is in agent a's sense region

        Args:
            agent_a (multiagent_particle_env.core.Agent): Agent object
            agent_b (multiagent_particle_env.core.Agent): Agent object

        Returns:
            (boolean) True if agent b in sense region of agent a else False
        """
        if not isinstance(agent_a, TurretAgent):
            # Compute actual distance between entities
            delta_pos = agent_a.state.p_pos - agent_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))

            # Minimum allowable distance
            dist_min = agent_a.size + agent_a.sense_region + agent_b.size

        else:
            # Compute actual distance between entities
            delta_pos = agent_a.state.p_pos - agent_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))

            # Minimum allowable distance
            dist_min = agent_a.sense_region + agent_b.size

        # Agent b in sense region of agent a if dist is less then the minimum allowable distance
        return True if dist < dist_min else False

    def get_collision_force(self, entity_a, entity_b):
        """
        Get collision forces for any contact between two entities

        Args:
            entity_a (multiagent_particle_env.Agent): An agent object.
            entity_b (multiagent_particle_env.Agent): An agent object.

        Returns:
            (list) All the agents in the world controllable by world scripts.
        """
        # An entity cannot collide with itself
        if entity_a is entity_b:
            return [None, None]

        # One or both of the entities are not colliders
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]

        # Compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        # Minimum allowable distance
        dist_min = entity_a.size + entity_b.size

        # Softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k

        # Collision force
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None

        return [force_a, force_b]
