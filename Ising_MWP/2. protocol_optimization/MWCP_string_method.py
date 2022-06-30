#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:41:25 2019

Code to calculate minimum-work protocols (MWPs) in multiple dimensions.
Numerically solves the Euler-Lagrange equation using the string method for the 
linear-reponse approximation of protocol excess work. Also, code to calculate 1D
optimized protocol for points equally spaced in time is included.

Citation:
string method - Rotskoff, Crooks, Vanden-Eijnden. "A geometric approach to 
optimal nonequilibrium control: Minimizing dissipation in nanomagnetic 
spin systems". PRE. (2017).
LR excess work - Sivak, Crooks. "Thermodynamic metrics and optimal paths." PRL. (2012).
Original reparameterization scheme - 

@author: MirandaLouwerse
"""
#load modules
import numpy as np
from string_parameters import *
import spline_4D as spl


# update string steps #

def update_coordinate(string,alpha,zeta_inv,zeta_deriv,R,delta_r,delta_t,num_blocks):
    #updates string using local friction information and the E-L equation
    
    string_velocity = np.gradient(string,axis=0)/delta_t
    
    matrix = np.zeros((R,R))
    rhs = np.zeros((R,1))
    
    for m in range(1,R-1):
        
        matrix[m,m-1] = -delta_r/(delta_t**2)
        matrix[m,m] = 1.0 + 2.0*delta_r/(delta_t**2)
        matrix[m,m+1] = -delta_r/(delta_t**2)
        
        to_add = 0.0
        for k in range(num_blocks):
            for j in range(num_blocks):
                for i in range(num_blocks):
                        
                    summand = zeta_inv[m][alpha,k]*string_velocity[m,i]*\
                        string_velocity[m,j]*(zeta_deriv[m][k,j,i]-0.5*zeta_deriv[m][i,j,k])
                    
                    to_add += summand
        
        rhs[m,0] = string[m,alpha] + delta_r*to_add
        
        m += 1 
    
    # fix the string endpoints
    matrix[0,0] = 1.0   
    rhs[0,0] = string[0,alpha]
    matrix[R-1,R-1] = 1.0
    rhs[R-1,0] = string[R-1,alpha]
    
    new_coord = np.linalg.solve(matrix,rhs)
    
    return new_coord


def update_coordinate_2D(string,alpha,zeta_inv,zeta_deriv,R,delta_r,delta_t,num_blocks):
    #updates string using local friction information and the E-L equation
    
    string_velocity = np.gradient(string,axis=0)/delta_t
    
    matrix = np.zeros((R,R))
    rhs = np.zeros((R,1))
    
    for m in range(1,R-1):
        
        matrix[m,m-1] = -delta_r/(delta_t**2)
        matrix[m,m] = 1.0 + 2.0*delta_r/(delta_t**2)
        matrix[m,m+1] = -delta_r/(delta_t**2)
        
        to_add = 0.0
        for k in [1,2]:
            for j in [1,2]:
                for i in [1,2]:
                        
                    summand = zeta_inv[m][alpha,k]*string_velocity[m,i]*\
                        string_velocity[m,j]*(zeta_deriv[m][k,j,i]-0.5*zeta_deriv[m][i,j,k])
                    
                    to_add += summand
        
        rhs[m,0] = string[m,alpha] + delta_r*to_add
        
        m += 1 
    
    # fix the string endpoints
    matrix[0,0] = 1.0   
    rhs[0,0] = string[0,alpha]
    matrix[R-1,R-1] = 1.0
    rhs[R-1,0] = string[R-1,alpha]
    
    new_coord = np.linalg.solve(matrix,rhs)
    
    return new_coord


def calc_smooth_string(string,R,smooth_parm):
    # takes an input string and smooths by averaging each string point with
    # its neighbours with weight of adjacent string points given by smooth_parm
    z_smooth = np.zeros((R,num_blocks))

    z_smooth[0] = string[0]
    z_smooth[-1] = string[-1]
    for m in range(1,R-1):
        z_smooth[m] = (1.0-smooth_parm)*string[m] + smooth_parm/2.0*(string[m-1]\
                    +string[m+1])

    return z_smooth


def avg_rot_string(string,R,num_blocks):
    # symmetrizes 180deg rotation in an input string
    
    rot_string = np.zeros((R,num_blocks))
    for i in range(R):
        rot_string[i,0] = 0.5*(string[i,0]-string[-i-1,0])
        rot_string[i,1] = 0.5*(string[i,1]-string[-i-1,2])
        rot_string[i,2] = 0.5*(string[i,2]-string[-i-1,1])
        rot_string[i,3] = 0.5*(string[i,3]-string[-i-1,3])
    
    return rot_string


def calculate_LR_comp_power(string,string_times,spline_fits):
    #calculates LR excess power in 4D for string using local friction information
    
    Rs = len(string)
    
    string_friction_list = []
    for m in range(Rs):
        friction,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(string[m],spline_fits,num_blocks)
        string_friction_list.append(friction)
    
    string_velocity = np.gradient(string,axis=0)*(Rs-1)
    
    LR_work = np.zeros((num_blocks))
    LR_power = np.zeros((Rs,num_blocks))
    for h1 in range(num_blocks):
        for h2 in range(num_blocks):            
            LR_work[h1] += string_velocity[0,h1]*string_friction_list[0][h1,h2]*\
                    string_velocity[0,h2]*(string_times[1]-string_times[0])/2.0
            LR_power[0,h1] += string_velocity[0,h1]*string_friction_list[0][h1,h2]*string_velocity[0,h2]
            for m in range(1,Rs-1):
                LR_work[h1] += string_velocity[m,h1]*string_friction_list[m][h1,h2]*\
                    string_velocity[m,h2]*(string_times[m+1]-string_times[m-1])/2.0
                LR_power[m,h1] += string_velocity[m,h1]*string_friction_list[m][h1,h2]*string_velocity[m,h2]
            LR_work[h1] += string_velocity[Rs-1,h1]*string_friction_list[Rs-1][h1,h2]*\
                    string_velocity[Rs-1,h2]*(string_times[Rs-1]-string_times[Rs-2])/2.0
            LR_power[Rs-1,h1] += string_velocity[Rs-1,h1]*string_friction_list[Rs-1][h1,h2]*string_velocity[Rs-1,h2]
    
    return LR_work,LR_power

def calculate_LR_comp_power_2D(string,string_times,spline_fits):
    #calculates LR excess power in 2D for string using local friction information
    
    Rs = len(string)
    
    string_friction_list = []
    for m in range(Rs):
        friction,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(string[m],spline_fits,num_blocks)
        string_friction_list.append(friction)
    
    string_velocity = np.gradient(string,axis=0)*(Rs-1)
    
    LR_work = np.zeros((num_blocks))
    LR_power = np.zeros((Rs,num_blocks))
    for h1 in [1,2]:
        for h2 in [1,2]:            
            LR_work[h1] += string_velocity[0,h1]*string_friction_list[0][h1,h2]*\
                    string_velocity[0,h2]*(string_times[1]-string_times[0])/2.0
            LR_power[0,h1] += string_velocity[0,h1]*string_friction_list[0][h1,h2]*string_velocity[0,h2]
            for m in range(1,Rs-1):
                LR_work[h1] += string_velocity[m,h1]*string_friction_list[m][h1,h2]*\
                    string_velocity[m,h2]*(string_times[m+1]-string_times[m-1])/2.0
                LR_power[m,h1] += string_velocity[m,h1]*string_friction_list[m][h1,h2]*string_velocity[m,h2]
            LR_work[h1] += string_velocity[Rs-1,h1]*string_friction_list[Rs-1][h1,h2]*\
                    string_velocity[Rs-1,h2]*(string_times[Rs-1]-string_times[Rs-2])/2.0
            LR_power[Rs-1,h1] += string_velocity[Rs-1,h1]*string_friction_list[Rs-1][h1,h2]*string_velocity[Rs-1,h2]
    
    return LR_work,LR_power

#functions for time optimization of 4D protocol
def L_sum_calculator_time(string,l,spline_fits):
    
    L_k = 0
    for m in range(1,l+1):
        friction_l,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(
            string[m],spline_fits,num_blocks)
        friction_l1,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(
            string[m-1],spline_fits,num_blocks)
        avg_friction = 0.5*(friction_l+friction_l1)
        
        distance = 0.0
        for i in range(num_blocks):
            for j in range(num_blocks):
                distance += (string[m,i]-string[m-1,i])*avg_friction[i,j]*(string[m,j]-string[m-1,j])
        L_k += distance
    
    return L_k


def reparameterize_point_time(string,m,spline_fits,R):
    # determine k
    
    s_m = L_sum_calculator_time(string,R-1,spline_fits)*float(m)/float(R-1)
    l=0
    L_k = 0
    while L_k < s_m:
        l += 1
        L_k = L_sum_calculator_time(string,l,spline_fits)
    
    # calculate the reparameterized string point
    L_k_1 = L_sum_calculator_time(string,l-1,spline_fits)
    friction_l,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(
            string[l],spline_fits,num_blocks)
    friction_l1,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(
            string[l-1],spline_fits,num_blocks)
    avg_friction = 0.5*(friction_l+friction_l1)
    distance = 0.0
    for i in range(num_blocks):
        for j in range(num_blocks):
            distance += (string[l,i]-string[l-1,i])*avg_friction[i,j]*(string[l,j]-string[l-1,j])
    z_new = np.zeros((num_blocks))
    
    for j in range(num_blocks):
        z_new[j] = string[l-1,j] + (s_m-L_k_1)/distance*(string[l,j] - string[l-1,j])

    return z_new


def ReparameterizeString_time(string,spline_fits,R):
    string_temp = np.zeros((R,num_blocks))
    
    for m in range(1,R-1):
        new_m_point = reparameterize_point_time(string,m,spline_fits,R)
        for j in range(num_blocks):
            string_temp[m,j] = new_m_point[j]
    
    for j in range(num_blocks):
        string_temp[0,j] = string[0,j]
        string_temp[R-1,j] = string[R-1,j]
    
    return string_temp


#functions for time optimization of 2D protocol
def L_sum_calculator_2D(string,l,spline_fits):
    
    L_k = 0
    for m in range(1,l+1):
        friction_l,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(
            string[m],spline_fits,num_blocks)
        friction_l1,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(
            string[m-1],spline_fits,num_blocks)
        avg_friction = 0.5*(friction_l+friction_l1)
        
        distance = 0.0
        for i in [1,2]:
            for j in [1,2]:
                distance += (string[m,i]-string[m-1,i])*avg_friction[i,j]*(string[m,j]-string[m-1,j])
        L_k += distance
    
    return L_k


def reparameterize_point_2D(string,m,spline_fits,R):
    # determine k
    
    s_m = L_sum_calculator_2D(string,R-1,spline_fits)*float(m)/float(R-1)
    l=0
    L_k = 0
    while L_k < s_m:
        l += 1
        L_k = L_sum_calculator_2D(string,l,spline_fits)
    
    # calculate the reparameterized string point
    L_k_1 = L_sum_calculator_2D(string,l-1,spline_fits)
    friction_l,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(
            string[l],spline_fits,num_blocks)
    friction_l1,friction_deriv = spl.interpolate_log_friction_and_deriv_spline(
            string[l-1],spline_fits,num_blocks)
    avg_friction = 0.5*(friction_l+friction_l1)
    distance = 0.0
    for i in [1,2]:
        for j in [1,2]:
            distance += (string[l,i]-string[l-1,i])*avg_friction[i,j]*(string[l,j]-string[l-1,j])
    z_new = np.zeros((num_blocks))
    
    for j in [1,2]:
        z_new[j] = string[l-1,j] + (s_m-L_k_1)/distance*(string[l,j] - string[l-1,j])

    return z_new


def ReparameterizeString_2D(string,spline_fits,R):
    string_temp = np.zeros((R,num_blocks))
    
    for m in range(1,R-1):
        new_m_point = reparameterize_point_2D(string,m,spline_fits,R)
        for j in [1,2]:
            string_temp[m,j] = new_m_point[j]
    
    for j in [1,2]:
        string_temp[0,j] = string[0,j]
        string_temp[R-1,j] = string[R-1,j]
    
    return string_temp
