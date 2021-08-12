# -*- coding: utf-8 -*-
"""
Created on Wed May 12 21:11:05 2021

@author: caiog
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data/Seasons/Winner_2008-2017.csv')
dataset = dataset.iloc[:,4:]
corrMatrix = dataset.corr()
print (corrMatrix)