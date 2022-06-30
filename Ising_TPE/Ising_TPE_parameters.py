#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:57:37 2020

Parameters for simulations of 4D protocols.

@author: MirandaLouwerse
"""

import numpy as np

# Parameters
J = 1.0
num_spins = 9
beta = 1.0
num_blocks = 4
spin_number = 9

A = 0
B = 511

comp_size = [4,2,2,1]

min_traj = 0
max_traj = 40000
num_traj = max_traj-min_traj

project_directory = "/Users/MirandaLouwerse/Documents/MirandaWork/research_files/connect_the_metrics"
save_directory = "{}/data/TPE_energy_flows".format(project_directory)
plot_directory = "{}/results/TPE_energy_flows".format(project_directory)
