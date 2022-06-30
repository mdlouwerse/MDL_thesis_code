#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:18:57 2020

This script contains functions to calculate 4D splines and interpolate values
and derivatives at arbitrary CP values within the CP space. This script is used
with all optimization scripts. Needs ARBInterp.py to calculate splines.

Citation: 

@author: MirandaLouwerse
"""

import numpy as np
import scipy.interpolate
import os
from string_parameters import *
from ARBInterp import quadcubic

# functions for importing and interpolation of log friction surface using splines

def get_smooth_log_spline_fits():
    #load friction surfaces, set negative values to fixed value, take log,
    #fit 4D cubic splines, save all splines to a list
    spline_fits = []
    for k in range(num_blocks**2):
        friction_k = np.loadtxt("{}/final_symmetrized_friction_{:.0f}.dat"
                               .format(friction_directory,k))
        
        for l in range(len(friction_k)):
            if friction_k[l,4] <= 0:
                friction_k[l,4] = 10**(-8)

        friction_k[:,4] = np.log(friction_k[:,4])

        spline_fits.append(quadcubic(friction_k))
        
    return spline_fits

def interpolate_log_friction_and_deriv_spline(h,spline_fits,num_blocks):
    #interpolate spline fits at CP value h and re-exponentiate value
    #to recover friction and derivative
    friction = np.zeros((num_blocks,num_blocks))
    friction_deriv = np.zeros((num_blocks,num_blocks,num_blocks))
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            k = i*num_blocks+j
            interpolation = spline_fits[k].Query2(h)
            friction[i,j] = np.exp(interpolation[0])
            for l in range(num_blocks):
                friction_deriv[i,j,l] = friction[i,j]*interpolation[1][l]
    
    return friction,friction_deriv
