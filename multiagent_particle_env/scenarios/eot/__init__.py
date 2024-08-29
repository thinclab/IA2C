# -*- coding: utf-8 -*-

import os.path as osp
import sys

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rolando.fernandez1.civ@army.mil'
__status__ = 'Dev'


def load(module_name):
    """
    Loads a python module from the path of the corresponding file.

    Args:
        module_name (str): Namespace where the python module will be loaded,
                           e.g. ``foo.bar``

    Returns:
        A valid module object

    Raises:
              ImportError: When the module can't be loaded
        FileNotFoundError: When module_path doesn't exist
    """
    module_path = osp.join(osp.dirname(__file__), module_name)

    # Python version 3.5 or greater
    if sys.version_info[0] == 3 and sys.version_info[1] >= 5:
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    # Python version 3.4.x or less
    elif sys.version_info[0] == 3 and sys.version_info[1] < 5:
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader(module_name, module_path)
        module = loader.load_module()
    # Python version 2
    # sys.version_info[0] == 2
    else:
        import imp
        module = imp.load_source(module_name, module_path)
    return module
