#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:08:55 2022

generate ensemble of transition paths from biased transition rates

@author: MirandaLouwerse
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random as rng
from random import randint
import Ising_TPE_functions as fn
from Ising_TPE_parameters import *

def Monte_Carlo_step(curr_state,committor,J,beta,num_spins):
    
    flip_state = np.zeros((num_spins))
    num = randint(0,num_spins-1)
    
    for j in range(num_spins):
        if j == num:
            flip_state[j] = -curr_state[j]
        else:
            flip_state[j] = curr_state[j]
    
    ham_next = fn.calc_hamiltonian(flip_state,J)
    ham_curr = fn.calc_hamiltonian(curr_state,J)
    q_next = committor[fn.get_index_from_state(flip_state,num_spins)]
    q_curr = committor[fn.get_index_from_state(curr_state,num_spins)]
    
    glauber_accept = 1.0/(1.0+np.exp(beta*(ham_next-ham_curr)))*q_next/q_curr #TPE transition rate
    
    num = rng(1)
    
    if num < glauber_accept:
        next_state = flip_state # accept the spin flip
    else:
        next_state = curr_state # reject the spin flip
    
    return next_state

def run_TPE_Monte_Carlo(A,B,committor,J,beta,num_spins):
    initial_state = select_initial_TPE_state(A,committor)
    
    state = initial_state
    traj = [fn.get_index_from_state(state,num_spins)]
    while fn.get_index_from_state(state,num_spins) != B:
        state = Monte_Carlo_step(state,committor,J,beta,num_spins)
        traj.append(fn.get_index_from_state(state,num_spins))
    
    return traj

def select_initial_TPE_state(A,committor):
    
    state_A = fn.get_state_from_index(A,num_spins)
    
    num = rng(1)
    
    cumulative_distr = np.zeros((num_spins))
    for i in range(num_spins):
        flip_state = np.zeros((num_spins))
        for j in range(num_spins):
            if j == i:
                flip_state[j] = -state_A[j]
            else:
                flip_state[j] = state_A[j]
        
        ham_next = fn.calc_hamiltonian(flip_state,J)
        ham_curr = fn.calc_hamiltonian(state_A,J)
        q_next = committor[fn.get_index_from_state(flip_state,num_spins)]
    
        glauber_accept = 1.0/(1.0+np.exp(beta*(ham_next-ham_curr)))*q_next/float(num_spins)
        
        if i == 0:
            cumulative_distr[i] = glauber_accept
        else:
            cumulative_distr[i] = glauber_accept + cumulative_distr[i-1]
    
    cumulative_distr = cumulative_distr/cumulative_distr[-1]
    
    i_next = 0
    while cumulative_distr[i_next] <= num:
        i_next += 1
    
    initial_state = np.zeros((num_spins))
    for j in range(num_spins):
        if j == i_next:
            initial_state[j] = -state_A[j]
        else:
            initial_state[j] = state_A[j]
    
    return initial_state

committor = fn.calc_committor(A,B)
for i in range(min_traj,max_traj):
    trajectory = run_TPE_Monte_Carlo(A,B,committor,J,beta,num_spins)

    np.save("{}/trajectories/traj_{:.0f}.npy".format(save_directory,i),trajectory)
