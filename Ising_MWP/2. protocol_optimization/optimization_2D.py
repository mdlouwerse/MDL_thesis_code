#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:24:07 2021

Code to calculate fully optimized protocol in 2D for 3x3 Ising model using
the string method. String is initialized from the naive protocol.

Citation: Louwerse, Sivak, "Multidimensional minimum-work
control of a 2D Ising model." J. Chem. Phys. (2022). 

@author: MirandaLouwerse
"""

import numpy as np
from string_parameters import *
import spline_4D as spl
import MWCP_string_method as sm

# program parameters
smooth_string = False    # determines whether to smooth string or not
smooth_till = 1000      # turn off string smoothing after this many iterations
smooth_parm = 0.2   # parameter for string smoothing. 0 is no smoothing, 1 maximizes smoothing
sym_string = True  # determines whether to symmetrize string
skip_plot = 10  # skip every skip_plot'th iteration in saving string
maxIteration = 15000    # maximum iteration to run the string
delta_r = 10**(-5)  # parameter for string update step
CP_list = [1,2]     # selects only blue and green components for 2D optimization

# where to save string
save_directory = "{}/data/paper_version/optimized_protocols/opt_2D".format(project_directory)

spline_fits = spl.get_smooth_log_spline_fits() # get friction surface interpolation

# Initialize string 
string_init = np.zeros((R,num_blocks))
init_rand = 0.0 # option to start string with a randomly-generated curvature.
for i in CP_list:
    for m in range(R):
        string_init[m,i] = m*(stringMax[i]-stringMin[i])/(R-1) + stringMin[i] +\
                (-1)**i*init_rand*m*(m-R+1)
string = string_init
iteration = 0

#calculate initial LR excess work and power
LR_work,LR_power = sm.calculate_LR_comp_power_2D(string,np.linspace(0,1,R),spline_fits)

# save string with LR excess work and power
np.save("{}/string_{:.0f}.npy".format(save_directory,iteration),string)
np.save("{}/power_{:.0f}.npy".format(save_directory,iteration),LR_power)
np.save("{}/work_{:.0f}.npy".format(save_directory,iteration),np.array([LR_work]))


# options to start running program from a string saved in a file
#iteration = 8000
#string = np.load("{}/string_{:.0f}.npy".format(save_directory,iteration))


# Loop string update steps
convergence_boolean = 1
while convergence_boolean == 1 and iteration <= maxIteration:
    iteration += 1
    
    # collect local friction information for each string point
    string_friction = [] #list of friction matrix at each string point
    string_deriv = [] #list of friction derivatives matrix at each string point
    string_inv = [] #list of friction matrix inverse at each string point
    for m in range(R):
        for j in range(num_blocks):
            if string[m,j] < hMin[j]+dh:
                string[m,j] = hMin[j]+dh+0.001
            if string[m,j] > hMax[j]-dh:
                string[m,j] = hMax[j]-dh-0.001
        
        # collect 4D friction and derivative
        friction,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(string[m],spline_fits,num_blocks)
        
        # define 2x2 friction from 4x4 matrix
        friction_2D = np.zeros((2,2))
        friction_2D[0,0] = friction[1,1]
        friction_2D[0,1] = friction[1,2]
        friction_2D[1,0] = friction[2,1]
        friction_2D[1,1] = friction[2,2]
        # take inverse of 2x2 matrix. 
        #pseudo-inverse is used to avoid errors for small determinants
        inv_friction_2D = np.linalg.pinv(friction_2D) 
        # keep 2x2 friction in 4x4 array for compatibility
        inv_friction = np.zeros((num_blocks,num_blocks))
        inv_friction[1,1] = inv_friction_2D[0,0]
        inv_friction[1,2] = inv_friction_2D[0,1]
        inv_friction[2,1] = inv_friction_2D[1,0]
        inv_friction[2,2] = inv_friction_2D[1,1]
        
        string_friction.append(friction)
        string_deriv.append(friction_deriv)
        string_inv.append(inv_friction)
        
    # update string
    string_update = np.zeros((R,num_blocks))
    for j in CP_list:
        update_coord = sm.update_coordinate_2D(string,j,string_inv,
                                string_deriv,R,delta_r,delta_t,num_blocks)
        string_update[:,j] = update_coord[:,0]
    
    # smooth the string
    if iteration > smooth_till:
        smooth_string = False
        
    if smooth_string == True:
        string_smooth = np.zeros((R,num_blocks))
        for m in range(1,R-1):
            new_m_point = sm.calc_smooth_string(string_update,m,smooth_parm)
            for j in CP_list:
                string_smooth[m,j] = new_m_point[0,j]

        string_smooth[0] = string_update[0]
        string_smooth[R-1] = string_update[R-1]
    else:
        string_smooth = string_update
    
    # symmetrize string points
    if sym_string == True:
        string = sm.avg_rot_string(string_smooth,R,num_blocks)
    else:
        string = string_smooth
    
    
    if iteration%skip_plot == 0:
        # check if any string points have left CP region
        # if so, put then back inside
        for m in range(R):
            for j in range(num_blocks):
                if string[m,j] < hMin[j]+dh: # splines don't work in edge bin region
                    string[m,j] = hMin[j]+dh+0.001
                if string[m,j] > hMax[j]-dh:
                    string[m,j] = hMax[j]-dh-0.001
        
        # save string iteration
        np.save("{}/string_{:.0f}.npy".format(save_directory,iteration),string)
        
        # calculate and save string LR excess work and power
        LR_work,LR_power = sm.calculate_LR_comp_power_2D(string,np.linspace(0,1,R),spline_fits)
        np.save("{}/power_{:.0f}.npy".format(save_directory,iteration),LR_power)
        np.save("{}/work_{:.0f}.npy".format(save_directory,iteration),np.array([LR_work]))
    
    # can add code here to check convergence criterion. If met, set convergence_boolean=0
