# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:19:38 2021

@author: caiog
"""

import pathlib
import sys

current_path = pathlib.Path(__file__).parent.resolve()
if (str(current_path) + '/machine_learning_models') not in sys.path:
    sys.path.append(str(current_path) + '/machine_learning_models')