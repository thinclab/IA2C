# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
logger.py
"""

import numpy as np
import errno
import os

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


class Log(object):
    def __init__(self, header):
        assert type(header) == list
        self.header = header
        self.data = []


class Logger(object):
    def __init__(self, is_logging):
        self.is_logging = is_logging
        if not self.is_logging:
            return
        self.logs = {}

    def new(self, key, header):
        if not self.is_logging:
            return
        self.logs[key] = Log(header)

    def add(self, log_id, data):
        if not self.is_logging:
            return
        log = self.logs[log_id]
        data = list(data)
        assert len(data) == len(log.header)

        self.logs[log_id].data.append(data)

    def save(self, log_id, path, filename=False, filetype="csv"):
        if not self.is_logging:
            return
        log = self.logs[log_id]
        try:
            path = path.split("/")
        except:
            path = path.split("\\")
        path = "/".join(path)
        if path[-1] != "/":
            path = path + "/"
        if not os.path.isdir(path):
            try:
                os.makedirs(path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        if not filename:
            filename = log_id
        if os.path.isfile(path + filename + "." + filetype):
            copy = 1
            new_filename = filename + "(" + str(copy) + ")"
            while os.path.isfile(path + new_filename + "." + filetype):
                copy += 1
                new_filename = filename + "(" + str(copy) + ")"
            filename = new_filename
        f = open(path + filename + "." + filetype, 'w')
        f.write(','.join([str(x) for x in log.header]) + "\n")
        for line in log.data:
            out_line = ','.join([str(x) for x in line]) + "\n"
            f.write(out_line)
        f.close()


def logger_test():
    """
    Simple test function for logging.

    Saves test log to current working directory.
    """
    test_logger = Logger(True)
    test_dir = r'{}/logger_test'.format(os.getcwd())
    header = ['x', 'y', 'z']
    test_logger.new('test', header)
    for i in range(10):
        data = np.random.random((3,))
        test_logger.add('test', data)
    test_logger.save('test', test_dir)
