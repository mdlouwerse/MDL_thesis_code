#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:24:33 2022

Functions to calculate transition path quantities for 3x3 Ising system
for arbitrary choice of A and B

@author: MirandaLouwerse
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from Ising_TPE_parameters import *

#system Hamiltonian
def calc_hamiltonian(sigma,J):
    
    ham = -J*(sigma[0]*sigma[1]+sigma[1]*sigma[2]+sigma[3]*sigma[4]+ # coupling interactions between internal spins
               sigma[4]*sigma[5]+sigma[6]*sigma[7]+sigma[7]*sigma[8]+
               sigma[0]*sigma[3]+sigma[3]*sigma[6]+sigma[1]*sigma[4]+
               sigma[4]*sigma[7]+sigma[2]*sigma[5]+sigma[5]*sigma[8])
    
    # coupling interactions between spins and boundary conditions
    # spins 3 and 5 have spin up conditions
    # spins 1 and 7 have spin down conditions
    ham += -J*(sigma[3]+sigma[5])-J*(-sigma[1]-sigma[7]) 
    
    return ham

#system conjugate force
def calc_conj_variable(sigma):
    conj_force = np.zeros((num_blocks))
    
    conj_force[0] = sigma[0]+sigma[2]+sigma[6]+sigma[8]
    conj_force[1] = sigma[1]+sigma[7]
    conj_force[2] = sigma[3]+sigma[5]
    conj_force[3] = sigma[4]
    
    return conj_force

#state vector to index
def get_index_from_state(state,spin_number):

    binary_string = ''
    for i in range(spin_number):
        if state[i] == -1:
            binary_string = binary_string + '0'
        else:
            binary_string = binary_string + '1'

    return int(binary_string,2)

#index to state vector
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

#acceptance probability
def flip_single_spin(current_state,spin_index):
    
    next_state = np.zeros((len(current_state)))
    for i in range(len(next_state)):
        next_state[i] = current_state[i]
    next_state[spin_index] = -next_state[spin_index]
    
    return next_state


def glauber_acceptance(curr_state,possible_state,beta,J):
    energy_current = calc_hamiltonian(curr_state,J)
    energy_proposed = calc_hamiltonian(possible_state,J)
    p_accept = 1.0/(1.0+np.exp(beta*(energy_proposed-energy_current)))
    
    return p_accept

#transition probability i -> j
def calc_transition_probability(index_i,index_j,beta,J,spin_number):
    state_i = get_state_from_index(index_i,spin_number)
    state_j = get_state_from_index(index_j,spin_number)
    
    if np.abs(state_i-state_j).sum() > 2:
        trans_prob = 0.0
    elif index_i == index_j:
        total_prob = 0.0
        for i in range(spin_number):
            next_state = flip_single_spin(state_i,i)
            total_prob += 1.0/float(spin_number)*glauber_acceptance(state_i,next_state,beta,J)
        trans_prob = 1.0 - total_prob
    else:
        trans_prob = 1.0/float(spin_number)*glauber_acceptance(state_i,state_j,beta,J)
    
    return trans_prob

#transition rate i -> j
def calc_transition_rate(index_i,index_j,beta,J,spin_number):
    state_i = get_state_from_index(index_i,spin_number)
    state_j = get_state_from_index(index_j,spin_number)
    
    if np.abs(state_i-state_j).sum() > 2:
        trans_rate = 0.0
    elif index_i == index_j:
        total_rate = 0.0
        for i in range(spin_number):
            next_state = flip_single_spin(state_i,i)
            total_rate += 1.0/float(spin_number)*glauber_acceptance(state_i,next_state,beta,J)
        trans_rate = -total_rate
    else:
        trans_rate = 1.0/float(spin_number)*glauber_acceptance(state_i,state_j,beta,J)
    
    return trans_rate

#transition rate matrix

#equilibrium probability vector
def calc_eqm_probability(beta,J,spin_number):
    partition_function = 0.0
    eqm_prob = np.zeros((2**spin_number))
    
    for i in range(2**spin_number):
        state_here = get_state_from_index(i,spin_number)
        ham_here = calc_hamiltonian(state_here,J)
        partition_function += np.exp(-beta*ham_here)
        eqm_prob[i] = np.exp(-beta*ham_here)
    
    return eqm_prob/partition_function

#equilibrium partition function
def calc_partition_function(beta,J,spin_number):
    partition_function = 0.0
    
    for i in range(2**spin_number):
        state_here = get_state_from_index(i,spin_number)
        ham_here = calc_hamiltonian(state_here,J)
        partition_function += np.exp(-beta*ham_here)
    
    return partition_function

#committor calculation
def calc_committor(A,B):
    committor_matrix = np.zeros((2**spin_number,2**spin_number))
    committor_rhs = np.zeros((2**spin_number))

    for i in range(2**spin_number):
        for j in range(2**spin_number):
            if i == A:
                continue
            if i == B:
                continue
            if i == j:
                trans_prob = calc_transition_probability(i,i,beta,J,spin_number)
                committor_matrix[i,j] = 1.0-trans_prob
            else:
                trans_prob = calc_transition_probability(i,j,beta,J,spin_number)
                committor_matrix[i,j] = -trans_prob
    committor_matrix[A,A] = 1.0
    committor_rhs[A] = 0.0
    committor_matrix[B,B] = 1.0
    committor_rhs[B] = 1.0
    
    committor = linalg.solve(committor_matrix,committor_rhs)
    
    return committor

#TPE transition rate matrix

#TPE probability vector
def calc_prob_reactive(committor):
    total_reactive_prob = 0
    prob_states = []
    
    for i in range(2**spin_number):
        state_here = get_state_from_index(i,spin_number)
        ham_here = calc_hamiltonian(state_here,J)
        total_reactive_prob += committor[i]*(1.0-committor[i])*np.exp(-beta*ham_here)
        prob_states.append(committor[i]*(1.0-committor[i])*np.exp(-beta*ham_here))
    
    prob_states = prob_states/total_reactive_prob
    
    return prob_states

#TPE probability flux matrix

#TPE energy flows
def all_works_each_field(committor):
    mag_list = [[-4,-2,0,2,4],[-2,0,2],[-2,0,2],[-1,1]]
    num_mag = [len(mag_list[0]),len(mag_list[1]),len(mag_list[2]),len(mag_list[3])]
    
    total_work = [np.zeros((num_mag[0]-1)),np.zeros((num_mag[1]-1)),
                       np.zeros((num_mag[2]-1)),np.zeros((num_mag[3]-1))]
    
    partition_function = calc_partition_function(beta,J,spin_number)
    for i in range(2**spin_number):
        for j in range(2**spin_number):
            state_curr = get_state_from_index(i,spin_number)
            state_next = get_state_from_index(j,spin_number)
            mag_curr = calc_conj_variable(state_curr)
            mag_next = calc_conj_variable(state_next)
            for k in range(num_blocks):
                m1 = mag_list[k].index(mag_curr[k])
                m2 = mag_list[k].index(mag_next[k])
                if np.abs(m2-m1) > 1.1:
                    continue
                elif m2 <= m1:
                    continue
                else:
                    ham_curr = calc_hamiltonian(state_curr,J)
                    ham_next = calc_hamiltonian(state_next,J)
                    q_curr = committor[i]
                    q_next = committor[j]
                    t_prob = calc_transition_probability(i,j,beta,J,spin_number)
                    eqm_curr = np.exp(-beta*ham_curr)/partition_function
                    
                    total_work[k][m1] += t_prob*eqm_curr*(q_next-q_curr)*(ham_next-ham_curr)
    
    return total_work

#TPE entropy production
def all_entropies_each_field(committor):
    mag_list = [[-4,-2,0,2,4],[-2,0,2],[-2,0,2],[-1,1]]
    num_mag = [len(mag_list[0]),len(mag_list[1]),len(mag_list[2]),len(mag_list[3])]
    
    total_entropy = [np.zeros((num_mag[0]-1)),np.zeros((num_mag[1]-1)),
                       np.zeros((num_mag[2]-1)),np.zeros((num_mag[3]-1))]
    
    total_reactive_prob = total_prob_reactive(committor)
    partition_function = calc_partition_function(beta,J,spin_number)
    for i in range(2**spin_number):
        for j in range(2**spin_number):
            state_curr = get_state_from_index(i,spin_number)
            state_next = get_state_from_index(j,spin_number)
            mag_curr = calc_conj_variable(state_curr)
            mag_next = calc_conj_variable(state_next)
            for k in range(num_blocks):
                m1 = mag_list[k].index(mag_curr[k])
                m2 = mag_list[k].index(mag_next[k])
                if np.abs(m2-m1) > 1.1:
                    continue
                elif m2 <= m1:
                    continue
                else:
                    ham_curr = calc_hamiltonian(state_curr,J)
                    ham_next = calc_hamiltonian(state_next,J)
                    q_curr = committor[i]
                    q_next = committor[j]
                    t_prob = calc_transition_probability(i,j,beta,J,spin_number)
                    eqm_curr = np.exp(-beta*ham_curr)/partition_function
                    
                    if q_curr == 0.0 or q_next == 0.0:
                        continue
                    elif q_next == 1.0 or q_curr == 1.0:
                        continue
                    else:
                        entropy_arg = np.log(((1.0-q_curr)*q_next)/(q_curr*(1.0-q_next)))
            
                    total_entropy[k][m1] += t_prob*eqm_curr*(q_next-q_curr)/total_reactive_prob*entropy_arg
    
    return total_entropy

#AB rate constant calculation
def calc_rate_constant_AB(committor):
    prob_A = 0.0

    prob_curr = 0.0
    for i in range(2**spin_number):
        for j in range(2**spin_number):
            state_curr = get_state_from_index(i,spin_number)
            state_next = get_state_from_index(j,spin_number)
            if np.abs(state_curr-state_next).sum() > 2:
                continue
            else:
                q_curr = committor[i]
                q_next = committor[j]
                t_prob = calc_transition_probability(i,j,beta,J,spin_number)
                eqm_curr = calc_eqm_probability(state_curr,beta,J,spin_number)
                
                prob_curr += 0.5*eqm_curr*t_prob*(q_next-q_curr)**2
                if i == j:
                    prob_A += eqm_curr*(1.0-q_curr)

    rate_constant = prob_curr/prob_A

    return rate_constant

def calc_rate_constant_BA(committor):
    prob_B = 0.0

    prob_curr = 0.0
    for i in range(2**spin_number):
        for j in range(2**spin_number):
            state_curr = get_state_from_index(i,spin_number)
            state_next = get_state_from_index(j,spin_number)
            if np.abs(state_curr-state_next).sum() > 2:
                continue
            else:
                q_curr = committor[i]
                q_next = committor[j]
                t_prob = calc_transition_probability(i,j,beta,J,spin_number)
                eqm_curr = calc_eqm_probability(state_curr,beta,J,spin_number)
                
                prob_curr += 0.5*eqm_curr*t_prob*(q_next-q_curr)**2
                if i == j:
                    prob_B += eqm_curr*q_curr

    rate_constant = prob_curr/prob_B

    return rate_constant

#AB free energy calculation
def calc_free_energy_AB(committor):
    k_AB = calc_rate_constant_AB(committor)
    k_BA = calc_rate_constant_BA(committor)
    
    return -np.log(k_AB/k_BA)

#TPE duration calculation
def calc_mean_TP_duration(committor):
    prob_react = total_prob_reactive(committor)

    prob_curr = 0.0
    partition_function = calc_partition_function(beta,J,spin_number)
    for i in range(2**spin_number):
        for j in range(2**spin_number):
            state_curr = get_state_from_index(i,spin_number)
            state_next = get_state_from_index(j,spin_number)
            if np.abs(state_curr-state_next).sum() > 2:
                continue
            else:
                q_curr = committor[i]
                q_next = committor[j]
                t_prob = calc_transition_probability(i,j,beta,J,spin_number)
                ham_curr = calc_hamiltonian(state_curr,J)
                eqm_curr = np.exp(-beta*ham_curr)/partition_function
                
                prob_curr += 0.5*eqm_curr*t_prob*(q_next-q_curr)**2

    mean_duration = prob_react/prob_curr

    return mean_duration

#RC length calculation

#TPE marginal probability
def total_prob_reactive(committor):
    #calculates total reactive prob*partition function
    total_reactive_prob = 0.0
    partition = 0.0
    
    for i in range(2**spin_number):
        state_here = get_state_from_index(i,spin_number)
        ham_here = calc_hamiltonian(state_here,J)
        partition += np.exp(-beta*ham_here)
        total_reactive_prob += committor[i]*(1.0-committor[i])*np.exp(-beta*ham_here)
        
    return total_reactive_prob/partition