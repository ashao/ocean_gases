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

toK = 273.15 # Conversion factor to absolute temperature

def check_tracer_name(gas):
    """Check to see if the named gas is recognized in this toolbox and return its name in this toolbox"""
    gas = gas.lower()
    if gas == 'cfc11' or gas == 'cfc-11' or gas == 'freon11' or gas == 'freon-11':
        return 'cfc11'
    elif gas == 'cfc12' or gas == 'cfc-12' or gas == 'freon12' or gas == 'freon-12':
        return 'cfc12'

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
    else:
        raise NameError("ocean_gases: %s is not a recognized gas" % gas)
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

def ttdmatch(measpcfc, meastime, gas, cfcatm, sat=1.):
    """
    Tries to find the inverse gaussian (with a fixed specified ratio) that matches the observed pCFC
    This is cast as a scalar optimization problem where gamma is varied until the difference between
    the observed pCFC and the convolution of the IG with the atmospheric history is minimized
    """
    def fmin(gamma, measpcfc, cfcinterp, tmax):
      # This is the function to be minimized: the difference between the observed partial
      # pressure and that done by convolving an IG with the atmospheric history
      conv, _ = quadrature(pdf_atm_expr, 0, tmax, args = (gamma, cfcinterp), tol = 1e-5)
      return np.sqrt((measpcfc - conv)**2)
    def pdf_atm_expr(t, gamma, cfcinterp):
      return inverse_gaussian(gamma, gamma, t)*cfcinterp(t)
    # Calculate the difference between the measured time and the atmospheric history curve
    reltime = [ meastime - t for t in cfcatm['year'][:] ]
    reltime = np.array([ t.total_seconds()/(86400*365.) for t in reltime ])
    # Build an interpolating function for the atmospheric history
    cfcinterp = interp1d(reltime, cfcatm[gas]*sat, fill_value = 0.,kind='quadratic')

    # Make sure that the value is reasonable
    if np.isnan(measpcfc) or measpcfc > np.max(cfcinterp.y) or measpcfc == 0.:
      return np.nan
    else:
      # Find the first time that the atmospheric history is 0 (prevents from doing quadrature over a bunch of 0s)
      for t in range(0, cfcinterp.y.size):
        if cfcinterp.y[t] == 0:
          break
      maxtime = cfcinterp.x[t]
      gamma,_ = minimize_scalar(fmin, args = (measpcfc, cfcinterp, maxtime), bounds = ( (0.1, 1000) ), method='bounded' ), reltime
      return gamma.x

