#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:24:07 2021

Code to calculate time-optimized protocols in 2D and 4D for 3x3 Ising model. 
A modified reparameterization scheme is used to make points equally spaced
in terms of friction matrix. Reparameterization code is in MWCP_string_method.py

Citation: Louwerse, Sivak, "Multidimensional minimum-work
control of a 2D Ising model." J. Chem. Phys. (2022). 

@author: MirandaLouwerse
"""

# import modules
import numpy as np
from string_parameters import *
import spline_4D as spl
import MWCP_string_method as sm

spline_fits = spl.get_smooth_log_spline_fits() # get friction surface interpolation

# initialize and save naive string in 4D

save_directory = "{}/data/paper_version/optimized_protocols/naive_4D".format(project_directory)

naive_4D = np.zeros((R,num_blocks))
for m in range(R):
    for i in range(num_blocks):
        naive_4D[m,i] = m*(stringMax[i]-stringMin[i])/(R-1) + stringMin[i]
LR_work,LR_power = sm.calculate_LR_comp_power(naive_4D,np.linspace(0,1,R),spline_fits)

#np.save("{}/control_protocol.npy".format(save_directory),naive_4D)
#np.save("{}/power.npy".format(save_directory),LR_power)
#np.save("{}/work.npy".format(save_directory),np.array([LR_work]))

# initialize and save naive string in 2D
save_directory = "{}/data/paper_version/optimized_protocols/naive_2D".format(project_directory)

naive_2D = np.zeros((R,num_blocks))
for m in range(R):
    for i in [1,2]:
        naive_2D[m,i] = m*(stringMax[i]-stringMin[i])/(R-1) + stringMin[i]
LR_work,LR_power = sm.calculate_LR_comp_power(naive_2D,np.linspace(0,1,R),spline_fits)

#np.save("{}/control_protocol.npy".format(save_directory),naive_2D)
#np.save("{}/power.npy".format(save_directory),LR_power)
#np.save("{}/work.npy".format(save_directory),np.array([LR_work]))

# time-optimize naive 4D string and save
save_directory = "{}/data/paper_version/optimized_protocols/time_4D".format(project_directory)

time_4D = sm.ReparameterizeString_time(naive_4D,spline_fits,R)

#np.save("{}/control_protocol.npy".format(save_directory),time_4D)
#np.save("{}/power.npy".format(save_directory),LR_power)
#np.save("{}/work.npy".format(save_directory),np.array([LR_work]))

# time-optimize naive 2D string and save
save_directory = "{}/data/paper_version/optimized_protocols/time_2D".format(project_directory)

time_2D = sm.ReparameterizeString_time(naive_2D,spline_fits,R)

#np.save("{}/control_protocol.npy".format(save_directory),time_2D)
#np.save("{}/power.npy".format(save_directory),LR_power)
#np.save("{}/work.npy".format(save_directory),np.array([LR_work]))
