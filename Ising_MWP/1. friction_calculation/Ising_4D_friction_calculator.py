#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:41:25 2019

Code to calculate friction matrix for 3x3 Ising model with 4 external field
control parameters.

@author: MirandaLouwerse
"""

import numpy as np
from numpy.random import random as rng
from random import randint
from Parameters import *

# basic equilibrium functions #

def calc_hamiltonian(sigma,h,J,num_blocks):
    # returns the hamiltonian of a system state under CP value h
    
    ham = -J*(sigma[0]*sigma[1]+sigma[1]*sigma[2]+sigma[3]*sigma[4]+ # coupling interactions between internal spins
               sigma[4]*sigma[5]+sigma[6]*sigma[7]+sigma[7]*sigma[8]+
               sigma[0]*sigma[3]+sigma[3]*sigma[6]+sigma[1]*sigma[4]+
               sigma[4]*sigma[7]+sigma[2]*sigma[5]+sigma[5]*sigma[8])
    
    # coupling interactions between spins and boundary conditions
    # spins 3 and 5 have spin up conditions
    # spins 1 and 7 have spin down conditions
    ham += -J*(sigma[3]+sigma[5])-J*(-sigma[1]-sigma[7]) 
    
    conj_force = calc_conj_force(sigma,num_blocks)
    for i in range(num_blocks): # external field contribution
        ham += -h[i]*conj_force[i] # choose -ve h so that same sign for field and spin decreases energy
    
    return ham


def get_state_from_index(index,spin_number):
    
    binary_string = '{0:b}'.format(index)
    if len(binary_string) < spin_number:
        binary_string = (spin_number-len(binary_string))*'0' + binary_string
    state_here = np.zeros((spin_number))

    for i in range(spin_number):
        if binary_string[i] == '0':
            state_here[i] = -1
        else:
            state_here[i] = 1
    
    return state_here


def calc_conj_force(sigma,num_blocks):
    conj_force = np.zeros((num_blocks))
    
    conj_force[0] = sigma[0]+sigma[2]+sigma[6]+sigma[8]
    conj_force[1] = sigma[1]+sigma[7]
    conj_force[2] = sigma[3]+sigma[5]
    conj_force[3] = sigma[4]
    
    return conj_force  


# Monte Carlo dynamics #

def Monte_Carlo_step(curr_state,h,J,beta,num_spins,num_blocks):
    
    flip_state = np.zeros((num_spins))
    num = randint(0,num_spins-1) #choose spin to flip
    
    for j in range(num_spins):
        if j == num:
            flip_state[j] = -curr_state[j]
        else:
            flip_state[j] = curr_state[j]
    
    ham_next = calc_hamiltonian(flip_state,h,J,num_blocks)
    ham_curr = calc_hamiltonian(curr_state,h,J,num_blocks)
    
    glauber_accept = 1.0/(1.0+np.exp(beta*(ham_next-ham_curr))) #acceptance criterion
    
    num = rng(1)
    
    if num < glauber_accept:
        next_state = flip_state # accept the spin flip
    else:
        next_state = curr_state # reject the spin flip
    
    return next_state


def integrate_correlation(ForceCorrelation,max_lag):

	Friction = 0

	#Numerically integrate the correlation function
	for index in range(max_lag):
		Friction += 0.5*(ForceCorrelation[index+1]+ForceCorrelation[index])

	return Friction


def select_initial_eqm_state(h,J,beta,num_blocks):
    
    num = rng(1)
    
    cumulative_distr = np.zeros((2**num_spins))
    state = get_state_from_index(0,num_spins)
    cumulative_distr[0] = np.exp(-beta*calc_hamiltonian(state,h,J,num_blocks))
    for i in range(1,2**num_spins):
        state = get_state_from_index(i,num_spins)
        cumulative_distr[i] = cumulative_distr[i-1]+np.exp(-beta*\
                        calc_hamiltonian(state,h,J,num_blocks))
    cumulative_distr = cumulative_distr/cumulative_distr[-1]
    
    i = 0
    while cumulative_distr[i] <= num:
        i += 1
    initial_state =  get_state_from_index(i,num_spins)
    
    return initial_state


def calc_point_friction(h,J,beta,trajectory_time,max_lag,num_spins,num_blocks):
    state = select_initial_eqm_state(h,J,beta,num_blocks)
    trajectory = [state]
    for i in range(trajectory_time):
        state = Monte_Carlo_step(state,h,J,beta,num_spins,num_blocks)
        trajectory.append(state)

    magnetization = []
    for j in range(num_blocks):
        magnetization.append([])
    
    for i in range(trajectory_time):
        for j in range(num_blocks):
            magnetization[j].append(calc_conj_force(trajectory[i],num_blocks)[j])
        
    meanMag = np.zeros((num_blocks))
    for j in range(num_blocks):
        meanMag[j] = np.mean(magnetization[j])
    
    DeltaMag = []
    for j in range(num_blocks):
        DeltaMag.append(magnetization[j] - meanMag[j])
    
    FFTMag = []
    for j in range(num_blocks):
        FFTMag.append(np.fft.rfft(DeltaMag[j],norm="ortho"))
    
    Friction = np.zeros((num_blocks,num_blocks))
    for j1 in range(num_blocks):
        for j2 in range(num_blocks):
            ForceCorr = np.fft.irfft(FFTMag[j1]*np.conj(FFTMag[j2]))
            Friction[j1,j2] = beta*integrate_correlation(ForceCorr,max_lag)

    return Friction

# Main Code #

for h0 in h0_list:
    for h1 in h1_list:
        for h2 in h2_list:
            for h3 in h3_list:
                h = np.array([h0,h1,h2,h3])
                friction = calc_point_friction(h,J,beta,
                           trajectory_time,max_lag,num_spins,num_blocks)
                np.save("{}/friction_{:.2f}_{:.2f}_{:.2f}_{:.2f}.npy"
                        .format(friction_directory, h0, h1, h2, h3), friction)
                
                
                