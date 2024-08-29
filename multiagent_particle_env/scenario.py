# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scenario.py

Contains the base skeleton for a scenario in the Multi-Agent Particle Environment

Updated and Enhanced version of OpenAI Multi-Agent Particle Environment
(https://github.com/openai/multiagent-particle-envs)
"""

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


class BaseScenario(object):
    """
    Defines the base scenario upon which the world is built.
    """
    def make_world(self, args):
        """
        Construct the world
        """
        raise NotImplementedError()

    def reset_world(self, world):
        """
        Reset the world to the initial conditions.
        """
        raise NotImplementedError()
