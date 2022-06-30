#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:38:07 2020

Runs a 4D protocol for durations in protocol_time_list for iterations
between min_traj and max_traj and outputs the trajectories as an array
to a subfolder in directory.

@author: MirandaLouwerse
"""

import numpy as np
from numpy.random import random as rng
from random import randint
from ProtocolParameters import *


# run Monte Carlo for protocol work estimation

def calc_hamiltonian(sigma,h,J,num_spins):
    
    ham = -J*(sigma[0]*sigma[1]+sigma[1]*sigma[2]+sigma[3]*sigma[4]+ # coupling interactions between internal spins
               sigma[4]*sigma[5]+sigma[6]*sigma[7]+sigma[7]*sigma[8]+
               sigma[0]*sigma[3]+sigma[3]*sigma[6]+sigma[1]*sigma[4]+
               sigma[4]*sigma[7]+sigma[2]*sigma[5]+sigma[5]*sigma[8])
    
    # coupling interactions between spins and boundary conditions
    # spins 3 and 5 have spin up conditions
    # spins 1 and 7 have spin down conditions
    ham += -J*(sigma[3]+sigma[5])-J*(-sigma[1]-sigma[7]) 
    
    ham += -h[0]*(sigma[0]+sigma[2]+sigma[6]+sigma[8])-h[1]*(sigma[1]+sigma[7])\
            -h[2]*(sigma[3]+sigma[5])-h[3]*sigma[4]
    
    return ham


def calc_conj_force(sigma,num_blocks):
    conj_force = np.zeros((num_blocks))
    
    conj_force[0] = sigma[0]+sigma[2]+sigma[6]+sigma[8]
    conj_force[1] = sigma[1]+sigma[7]
    conj_force[2] = sigma[3]+sigma[5]
    conj_force[3] = sigma[4]
    
    return conj_force


def get_state_from_index(index,spin_number):
    
    binary_string = '{0:b}'.format(index)
    if len(binary_string) < spin_number:
        binary_string = (spin_number-len(binary_string))*'0' + binary_string
    state = np.zeros((spin_number))

    for i in range(spin_number):
        if binary_string[i] == '0':
            state[i] = -1
        else:
            state[i] = 1
    
    return state


def Monte_Carlo_step(curr_state,h,J,beta,num_spins):
    
    flip_state = np.zeros((num_spins))
    num = randint(0,num_spins-1)
    
    for j in range(num_spins):
        if j == num:
            flip_state[j] = -curr_state[j]
        else:
            flip_state[j] = curr_state[j]
    
    ham_next = calc_hamiltonian(flip_state,h,J,num_spins)
    ham_curr = calc_hamiltonian(curr_state,h,J,num_spins)
    
    glauber_accept = 1.0/(1.0+np.exp(beta*(ham_next-ham_curr)))
    
    num = rng(1)
    
    if num < glauber_accept:
        next_state = flip_state # accept the spin flip
    else:
        next_state = curr_state # reject the spin flip
    
    return next_state


def h_at_time(time,protocol_time,string,string_times):
    
    scaled_time = float(time)/float(protocol_time)
    
    if scaled_time <= 0.0:
        h = string[0,:]
    
    for i in range(len(string_times)):
        if string_times[i] < scaled_time:
            h = np.zeros((num_blocks))
            for j in range(num_blocks):
                h[j] = (string[i+1,j]-string[i,j])*(scaled_time-string_times[i])\
                    /(string_times[i+1]-string_times[i])+string[i,j]
    
    return h


def select_initial_eqm_state(h,J,beta,num_spins):
    
    num = rng(1)
    
    cumulative_distr = np.zeros((2**num_spins))
    state = get_state_from_index(0,num_spins)
    cumulative_distr[0] = np.exp(-beta*calc_hamiltonian(state,h,J,num_spins))
    for i in range(1,2**num_spins):
        state = get_state_from_index(i,num_spins)
        cumulative_distr[i] = cumulative_distr[i-1]+np.exp(-beta*\
                        calc_hamiltonian(state,h,J,num_spins))
    cumulative_distr = cumulative_distr/cumulative_distr[-1]
    
    i = 0
    while cumulative_distr[i] <= num:
        i += 1
    initial_state =  get_state_from_index(i,num_spins)
    
    return initial_state


def run_neq_Monte_Carlo(string,string_times,J,beta,num_spins,equil_time,protocol_time):
    initial_state = select_initial_eqm_state(string[0],J,beta,num_spins)
    
    state = initial_state
    traj = np.zeros((protocol_time+1,num_spins))
    for i in range(protocol_time):
        h_here = h_at_time(i,protocol_time,string,string_times)
        state = Monte_Carlo_step(state,h_here,J,beta,num_spins)
        
        for j in range(num_spins):
            traj[i,j] = state[j]
    
    return traj



string = np.load("{}/control_protocol.npy".format(string_directory))
string_times = np.linspace(0,1,len(string))

for protocol_time in protocol_time_list:
    for i in range(min_traj,max_traj):
        trajectory = run_neq_Monte_Carlo(string,string_times,J,beta,num_spins,equilibration_time,protocol_time)
        np.save("{}/duration_{:.0f}/trajectories/traj_{:.0f}.npy".format(string_directory,protocol_time,i),trajectory)
