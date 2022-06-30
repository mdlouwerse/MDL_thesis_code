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

protocol_time_list = [1,10,50,100,500,1000,2000,5000,10000,20000,50000,100000]
string_list = ["naive_4D","naive_2D","time_4D","time_2D","opt_4D","opt_2D"]
comp_size = [4,2,2,1]

equilibration_time = 5000
min_traj = 0
max_traj = 5000
num_traj = max_traj-min_traj

project_directory = "/Users/MirandaLouwerse/Documents/MirandaWork/research_files/Ising_model"
string_directory = "{}/data/paper_version/\
optimized_protocols/{}".format(project_directory,string_list[0])
save_directory = "{}/data/paper_version/\
protocol_analysis/{}".format(project_directory,string_list[0])
analyze_directory = "{}/data/paper_version/\
protocol_analysis".format(project_directory)
