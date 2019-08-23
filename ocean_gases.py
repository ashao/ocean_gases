#!/usr/bin/env python
"""
ocean_gases - Toolbox for dealing with trace gases like CFC-11, CFC-12, and SF6
---
Function descriptions:
    tracer_sol:        Given salinity, temperature, return the solubility of a gas
    tracer_sol_coeffs: Return solubility for given gas in either volumetric or gravimetric units
    tracer_conc_to_pp: Given a salinity, temperature, and concentration, return the partial pressure
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.integrate import quadrature

toK = 273.15 # Conversion factor to absolute temperature
atmhist = pd.read_csv('CFC_atmospheric_histories_revised_2015_Table1.csv',skiprows=[1])

def check_tracer_name(gas):
    """Check to see if the named gas is recognized in this toolbox and return its name in this toolbox"""
    gas = gas.lower()
    cfc11_names = ['cfc11','cfc-11','freon11','freon-11','freon_11']
    cfc12_names = ['cfc12','cfc-12','freon12','freon-12','freon_12']
    sf6_names   = ['sf6','sulfur_hexaflouride']

    if gas in cfc11_names:
        return 'cfc11'
    elif gas in cfc12_names:
        return 'cfc12'
    elif gas in sf6_names:
        return 'sf6'
    else:
        raise NameError(f'ocean_gases: {gas} is not a recognized gas')

def tracer_sol_coeffs(gas, units = 'gravimetric'):
    """
    Returns the solubility coefficients associated with a given gas

    """
    gas = check_tracer_name(gas)
    if gas == 'cfc11': # Warner and Weiss, DSR 1985
        if 'grav' in units:
            a = [-232.0411, 322.5546, 120.4956, -1.39165]
            b = [-0.146531, 0.093621, -0.0160693]
        elif 'vol' in units:
            a = [-229.9261, 319.6552, 119.4471, -1.39165]
            b = [-0.142382, 0.091459, -0.0157274]
    elif gas == 'cfc12': # Warner and Weiss, DSR 1985
        if 'grav' in units:
            a = [-220.2120, 301.8695, 114.8533, -1.39165]
            b = [-0.147718, 0.093175, -0.0157340]
        elif 'vol' in units:
            a = [-218.0971, 298.9702, 113.8049, -1.39165]
            b = [-0.143566, 0.091015, -0.0153924]
    elif gas == 'sf6': # Bullister 2002, DSR 
        if 'grav' in units:
            a = [-82.1639, 120.152, 30.6372, 0.]
            b = [0.0293201, -0.0351974, 0.00740056]
        elif 'vol' in units:
            a = [-80.0343, 117.232, 29.5817, 0.]
            b = [0.0335183, -0.0373942, 0.00774862]
    return a, b

def tracer_sol(S, T, gas, units = 'gravimetric'):
    """
    Return the solubility of a gas given T, S, gas type, and desired units
        S:          salinity
        T:          temperature
        gas:        cfc11 or cfc12 and variants thereof
        units:      gravimetric(default) or volumetric
    """
    gas = check_tracer_name(gas)
    a, b = tracer_sol_coeffs(gas, units)
    if gas == 'cfc11' or gas=='cfc12':
        T = T + toK
        F = a[0] + a[1]*(100./T) + a[2]*np.log(T*0.01) + a[3]*(T*0.01)**2 + S*(b[0] + b[1]*(T*0.01) + b[2]*(T*0.01)**2)
        return np.exp(F)

def tracer_conc_to_pp(S, T, conc, gas, units = 'gravimetric'):
    """
    Converts a concentration to a partial pressure
        S:          Salinity
        T:          Temeprature
        conc:       Concentration
        gas:        cfc11 or cfc12 and variants thereof
        units:      gravimetric(default) or volumetric
    """
    sol = tracer_sol(S, T, gas, units)
    return conc/sol

def inverse_gaussian(gamma, delta, t):
    """
    Return the PDF of an inverse gaussian:
        gamma: Mean age
        delta: Width parameter
        t:     Times at which to evaluate the pdf
    """
    pdf = np.zeros(t.shape)
    pdf[t>0] = np.sqrt( gamma**3 / (4.*np.pi*delta**2*t[t>0]**3) ) * np.exp( (-gamma*(t[t>0]-gamma)**2) / (4*delta**2*t[t>0]) )
    return pdf

def ttdmatch(measpcfc, meastime, gascol, sat=1.):
    """
    Tries to find the inverse gaussian (with a fixed specified ratio) that matches the observed pCFC
    This is cast as a scalar optimization problem where gamma is varied until the difference between
    the observed pCFC and the convolution of the IG with the atmospheric history is minimized
    Inputs:
      measpcfc (float, partial pressure): The measured gas value
      meastime (float, year fraction)   : The time in the form of year fraction (e.g. 1980.5)
      gascol   (string)                 : Column name for the gas (e.g. CFC11NH)
      sat      (flloat, nondim)         : An an assumed saturation value
    """
    def fmin(gamma, measpcfc, cfcinterp, tmax):
      # This is the function to be minimized: the difference between the observed partial
      # pressure and that done by convolving an IG with the atmospheric history
      conv, _ = quadrature(pdf_atm_expr, 0, tmax, args = (gamma, cfcinterp), tol=0.1)
      return np.sqrt((measpcfc - conv)**2)
    def pdf_atm_expr(t, gamma, cfcinterp):
      return inverse_gaussian(gamma, gamma, t)*cfcinterp(t)
    # Calculate the difference between the measured time and the atmospheric history curve
    # Convert from yearfraction to a datetime object
    yearbase = atmhist['Year'].values.astype(int) # Base year
    yearfrac = atmhist['Year'].values - yearbase  # Fractional part
    days_in_year = [datetime(year+1,1,1) - datetime(year,1,1) for year in yearbase]
    year_datenum = [datetime(yearbase[i],1,1) + yearfrac[i]*(days_in_year)[i] for i in range(len(yearbase))]
    year_datenum = [np.datetime64(date.isoformat()) for date in year_datenum]
    reltime = (meastime - year_datenum).astype(float)/(1e9*365*86400)
    # Build an interpolating function for the atmospheric history
    cfcinterp = interp1d(reltime, atmhist[gascol].values*sat, fill_value = 0.,kind='quadratic')
    # Make sure that the value is reasonable
    if np.isnan(measpcfc) or measpcfc > np.max(cfcinterp.y) or measpcfc <= 0.:
      return np.nan
    else:
      # Find the first time that the atmospheric history is 0 (prevents from doing quadrature over a bunch of 0s)
      for t in range(0, cfcinterp.y.size):
        if cfcinterp.y[t] == 0:
          break
      maxtime = cfcinterp.x[t]
      gamma,_ = minimize_scalar(fmin, args = (measpcfc, cfcinterp, maxtime), bounds = ( (0.1, 1000) ), method='bounded'), reltime
      return gamma.x

