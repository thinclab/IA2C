# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.

Can be called by using, for example:
    env = make_env('simple_speaker_listener')

After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

from multiagent_particle_env.environment import MultiAgentEnv
from multiagent_particle_env.logger import Log, Logger

import multiagent_particle_env.scenarios as scenarios

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


def make_env(scenario_name, arglist=None, done=False, logging=False, benchmark=False):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().

    Use env.render() to view the environment on the screen.

    Some useful environment properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents

    Args:
        scenario_name (string): Name of the scenario from ./scenarios/ to be Returns
                                (without the .py extension)
        arglist (argparse.Namespace): Parsed commandline arguments object
        done (boolean): Whether the scenario uses a done function
        logging (boolean): Whether you want to produce logging data
                           (usually only done during evaluation)
        benchmark (boolean): Whether you want to produce benchmarking data
                             (usually only done during evaluation)

    Returns:
        env (multiagent_particle_env.environment.MultiAgentEnv): Multi-Agent Particle Environment object
    """
    # Load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    # Create world
    if arglist is not None:
        world = scenario.make_world(arglist)
    else:
        world = scenario.make_world()

    # Set up logger
    logger = Logger(logging)

    # Set up callbacks
    info_callback = None
    logging_callback = None
    done_callback = None

    if done:
        done_callback = scenario.done

    if logging:
        logging_callback = scenario.logging

    if benchmark:
        info_callback = scenario.benchmark_data

    if hasattr(world, 'stag_hunt') or hasattr(world, 'turret_cg'):
        world.scenario = scenario
        info_callback = scenario.info

    # Create multi-agent environment
    env = MultiAgentEnv(world, arglist, logger, reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward, observation_callback=scenario.observation,
                        logging_callback=logging_callback, info_callback=info_callback, done_callback=done_callback)

    return env
