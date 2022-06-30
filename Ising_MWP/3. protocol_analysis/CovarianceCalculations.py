#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:40:34 2021

Calculates equilibrium spin-spin covariance for each protocol and saves to a file

@author: MirandaLouwerse
"""

import numpy as np
import matplotlib.pyplot as plt

project_directory = "/Users/MirandaLouwerse/Documents/MirandaWork/research_files/Ising_model"
analyze_directory = "{}/data/paper_version/protocol_analysis".format(project_directory)
plot_directory = "{}/results/paper_version".format(project_directory)

J = 1.0
beta = 1.0
num_blocks = 4
num_spins = 9
string_list = ['naive_2D','time_2D','opt_2D','naive_4D','time_4D','opt_4D']
vert_string_list =  ["naive_2D","naive_4D","time_2D","time_4D","opt_2D","opt_4D"]

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


def calc_conj_force(sigma,num_blocks):
    
    conj_force = np.zeros((num_blocks))
    conj_force[0] = sigma[0]+sigma[2]+sigma[6]+sigma[8]
    conj_force[1] = sigma[1]+sigma[7]
    conj_force[2] = sigma[3]+sigma[5]
    conj_force[3] = sigma[4]

    return conj_force  


def calculate_freeenergy(h,J,beta,num_blocks):
    # returns the second cumulant for CP value h
    
    partition_function = 0.0
    
    for s0 in [-1,1]:
        for s1 in [-1,1]:
            for s2 in [-1,1]:
                for s3 in [-1,1]:
                    for s4 in [-1,1]:
                        for s5 in [-1,1]:
                            for s6 in [-1,1]:
                                for s7 in [-1,1]:
                                    for s8 in [-1,1]: # sum over all 2^N states of the system
                                        state = np.array([s0,s1,s2,s3,s4,s5,s6,s7,s8])
                                        hamiltonian = calc_hamiltonian(state,h,J,num_blocks)
                                    
                                        partition_function += np.exp(-beta*hamiltonian)
    
    return -np.log(partition_function)  


def calculate_meanforce(h,J,beta,num_blocks):
    # returns the first cumulant for CP value h
    
    partition_function = 0.0
    first_moment = np.zeros((num_blocks))
    
    for s0 in [-1,1]:
        for s1 in [-1,1]:
            for s2 in [-1,1]:
                for s3 in [-1,1]:
                    for s4 in [-1,1]:
                        for s5 in [-1,1]:
                            for s6 in [-1,1]:
                                for s7 in [-1,1]:
                                    for s8 in [-1,1]: # sum over all 2^N states of the system
                                        state = np.array([s0,s1,s2,s3,s4,s5,s6,s7,s8])
                                        hamiltonian = calc_hamiltonian(state,h,J,num_blocks)
                                        conj_force = calc_conj_force(state,num_blocks)
                                    
                                        partition_function += np.exp(-beta*hamiltonian) # add contribution of state to partition function
                                        for i1 in range(num_blocks):
                                            first_moment[i1] += conj_force[i1]*np.exp(-beta*hamiltonian)
    
    first_cumulant = first_moment/partition_function
    
    return first_cumulant  


def calculate_forcevar(h,J,beta,num_blocks):
    # returns the second cumulant for CP value h
    
    partition_function = 0.0
    first_moment = np.zeros((num_blocks))
    second_moment = np.zeros((num_blocks,num_blocks))
    
    for s0 in [-1,1]:
        for s1 in [-1,1]:
            for s2 in [-1,1]:
                for s3 in [-1,1]:
                    for s4 in [-1,1]:
                        for s5 in [-1,1]:
                            for s6 in [-1,1]:
                                for s7 in [-1,1]:
                                    for s8 in [-1,1]: # sum over all 2^N states of the system
                                        state = np.array([s0,s1,s2,s3,s4,s5,s6,s7,s8])
                                        hamiltonian = calc_hamiltonian(state,h,J,num_blocks)
                                        conj_force = calc_conj_force(state,num_blocks)
                                    
                                        partition_function += np.exp(-beta*hamiltonian) # add contribution of state to partition function
                                        for i1 in range(num_blocks):
                                            first_moment[i1] += conj_force[i1]*np.exp(-beta*hamiltonian)
                                            for i2 in range(num_blocks):
                                                second_moment[i1,i2] += conj_force[i2]*conj_force[i1]*np.exp(-beta*hamiltonian)
    
    first_cumulant = first_moment/partition_function
    second_cumulant = np.zeros((num_blocks,num_blocks))
    
    for i1 in range(num_blocks):
        for i2 in range(num_blocks):
            second_cumulant[i1,i2] = second_moment[i1,i2]/partition_function\
                -first_cumulant[i1]*first_cumulant[i2]
    
    return second_cumulant  

def calculate_spinvar(h,J,beta,num_spins):
    # returns the second cumulant for CP value h
    
    partition_function = 0.0
    first_moment = np.zeros((num_spins))
    second_moment = np.zeros((num_spins,num_spins))
    
    for s0 in [-1,1]:
        for s1 in [-1,1]:
            for s2 in [-1,1]:
                for s3 in [-1,1]:
                    for s4 in [-1,1]:
                        for s5 in [-1,1]:
                            for s6 in [-1,1]:
                                for s7 in [-1,1]:
                                    for s8 in [-1,1]: # sum over all 2^N states of the system
                                        state = np.array([s0,s1,s2,s3,s4,s5,s6,s7,s8])
                                        hamiltonian = calc_hamiltonian(state,h,J,num_blocks)
                                    
                                        partition_function += np.exp(-beta*hamiltonian) # add contribution of state to partition function
                                        for i1 in range(num_spins):
                                            first_moment[i1] += state[i1]*np.exp(-beta*hamiltonian)
                                            for i2 in range(num_spins):
                                                second_moment[i1,i2] += state[i2]*state[i1]*np.exp(-beta*hamiltonian)
    
    first_cumulant = first_moment/partition_function
    second_cumulant = np.zeros((num_spins,num_spins))
    
    for i1 in range(num_spins):
        for i2 in range(num_spins):
            second_cumulant[i1,i2] = second_moment[i1,i2]/partition_function\
                -first_cumulant[i1]*first_cumulant[i2]
    
    return second_cumulant  


for file_string in string_list:
    string = np.load("{}/{}/control_protocol.npy".format(analyze_directory,file_string))
    R = len(string)
    dt = float(1)/float(R-1)

    string_spinvar = np.zeros((R,num_spins,num_spins))
    for i in range(R):
        h = string[i]
    
        spinvar_here = calculate_spinvar(h,J,beta,num_spins)
    
        string_spinvar[i,:,:] = spinvar_here
    np.save("{}/{}/string_spinvariance.npy".format(analyze_directory,file_string),string_spinvar)
