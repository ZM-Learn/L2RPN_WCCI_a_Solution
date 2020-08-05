# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:12:39 2020

@author: yanzm
"""

import numpy as np

loaded = np.load('actions.npz')
actions_recording = (loaded['data'])  # this has 157 actions
