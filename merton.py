import numpy as np
from scipy.stats import norm

ln = np.log
sqrt = np.sqrt
exp = np.exp

params = {
        'V': 250.19, 
        'F': 117.83,
        'r': 0.05,
        'sig': 0.28,
        'T': 8.,
        't': 0. ,
        'sig_debt': 0
        }

def N(x):
    return norm.cdf(x)

def merton(params):
    V = params['V']
    F = params['F']
    r = params['r']
    T = params['T']
    t = params['t']
    sig = params['sig']

    d1 = ln(V/F) + (r + sig**2/2)*(T-t)
    d1 /= sig*sqrt(T-t)
    d2 = ln(V/F) + (r - sig**2/2)*(T-t)
    d2 /= sig*sqrt(T-t)

    D = F*exp(-r*(T-t))*N(d2) + V*N(-d1)
    E = V - D

    tmp = N(-d1) + F*exp(-r*(T-t))/V * N(d2)
    sig_debt = N(-d1) / tmp * sig
    return D, E, sig_debt

def find_asset_sigma(sig_debt_orig):
    for sig in np.linspace(0,1,41):
        params['sig'] = sig 
        D, E, sig_debt = merton(params)
        print(round(sig,2), abs(round(sig_debt-sig_debt_orig,4) ))


def find_asset_sigma_de(de_ratio_obs):
    for sig in np.linspace(0,0.1,41):
        params['sig'] = sig 
        params['r'] = 0.05
        D, E, sig_debt = merton(params)
        # print(D,E)
        de_rat_cal = D/E
        print(round(sig,2), de_ratio_obs, de_rat_cal)

D, E, sig_debt = merton(params)
print(D,E,sig_debt,D/E)
