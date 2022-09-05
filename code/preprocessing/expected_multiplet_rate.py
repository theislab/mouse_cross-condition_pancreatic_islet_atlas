# Set of helper functions for calculating expected multiplet rate.

import numpy as np
from scipy.optimize import fsolve


def solve_mu(mu:float,doublet_rate:float):
    return [(1+mu[0])*np.exp(-mu[0])-(1-doublet_rate)]

def get_mu(doublet_rate:float=0.1):
    return fsolve(solve_mu, [0.5],args=(doublet_rate))[0]

MU=get_mu()

def solve_N(N:float, Ns:list, mu:float=MU):
    res=1
    for Ni in Ns:
        res=res*(1-(Ni/N[0]))
    res=res-np.exp(-mu)
    return [res]

def get_N(Ns:list,mu:float=MU):
    return fsolve(solve_N, [10000],args=(Ns,mu))[0]


def mu_cell_type(N_cell_type:float, N:float):
    return -np.log((N-N_cell_type)/N)


def expected_multiplet(mu_present:list, mu_absent:list, N:float):
    res=N
    for mui in mu_present:
        res=res*(1-np.exp(-mui))
    for mui in mu_absent:
        res=res*np.exp(-mui)
    return res

