#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:09:06 2019

@author: MirandaLouwerse
"""

import numpy as np

# system parameters #

beta = 1.0
J = 1.0
num_spins = 9 # N, the number of spins in the system
num_blocks = 4 # M, the number of blocks of fields in the system

# control parameter grid parameters #
hMin = [-2.1,-3.0,-1.0,-2.1] # minimum bound on CP space for each CP component
hMax = [2.1,1.0,3.0,2.1] # maximum bound on CP space for each CP component
grid_num = [22,21,21,22] # number of discrete grid points in each CP direction
dh = 0.2 # spacing between discrete grid points

# lists of discrete grid points for each CP component
h0_list = np.round(np.linspace(hMin[0],hMax[0],grid_num[0]),1) 
h1_list = np.round(np.linspace(hMin[1],hMax[1],grid_num[1]),1)
h2_list = np.round(np.linspace(hMin[2],hMax[2],grid_num[2]),1)
h3_list = np.round(np.linspace(hMin[3],hMax[3],grid_num[3]),1)

# trajectory and friction calculation parameters #
trajectory_time = 10000000
max_lag = 25000  

# change directory file paths
project_directory = "/Users/MirandaLouwerse/Documents/MirandaWork/research_files/Ising_model"
friction_directory = "{}/data/paper_version/friction_estimates/run1".format(project_directory)
