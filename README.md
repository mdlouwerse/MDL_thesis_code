# thesis_code
 Code supporting work in Miranda Louwerse's PhD thesis

Code to calculate friction at a control parameter value, optimize control protocols, and run and analyze protococol simulations for a 3x3 Ising model is available in the Ising_MWP directory.

1. friction calculations

Ising_4D_friction_calculator.py:
Simulates long trajectory of spin-flip dynamics and calculates 4x4 generalized friction at one control-parameter value.

Parameters.py
Parameters for generalized friction calculation.

2. protocol optimization

string_parameters.py:
Parameters for protocol optimization

spline_4D.py:
Calculates 4D splines and evaluates interpolated value and derivative at a given 4D control parameter value.

ARBInterp.py:
Build splines. Citation:

MWCP_string_method.py:
Functions for calculating optimized protocols using the string method.

optimization_1D.py:
Calculates 2D and 4D time optimized protocols. A reparameterization scheme (cite:) is used to give control parameter values at evenly spaced times that are equally spaced in terms of the 1D generalized friction. 

optimization_2D.py:
Calculates 2D fully optimized protocol using the string method.

optimization_4D.py:
Calculates 4D fully optimized protocol using the string method.

3. protocol simulation and analysis

ProtocolParameters.py:
Parameters for running and analyzing protocols.

protocol_run.py:
Simulates system response to driving protocol for a given duration.

protocol_analyze.py:
Analyzes simulated protocol response and obtains ensemble-average properties.

CovarianceCalculations.py:
Calculates equilibrium spin-spin covariance for a given control protocol.


Code to generate an ensemble of transition paths using TPE transition rates for a 3x3 Ising model is available in the Ising_TPE directory.

Ising_TPE_functions.py:
Functions to calculate various TPE properties for 3x3 Ising model.

Ising_TPE_parameters.py:
Parameters for TPE trajectory simulation.

Ising_TPE_traj_generator.py:
Simulates trajectories from the TPE using modified transition rates.
