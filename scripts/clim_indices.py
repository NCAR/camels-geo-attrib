#!/usr/bin/env python

### PURPOSE

# Collection of the scripts to compute climatic indices.
# For some indices, several formulations have been implemented and the resulting estimates are returned as a
# data.frame. Alternative formulations can be added. The objective is to assess the sensitvity of the results to
# the formulation of the climatic indices.

# custom unctions can be coded for one location and can be applied to all the hrus with xarray apply_ufunc to broadcast the function
# seasonality_index, high_p_freq_dur, low_p_freq_dur

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime
import xarray as xr
from itertools import groupby
import multiprocessing as mp
from functools import partial
from sklearn.utils.extmath import weighted_mode
import time

# count consecutive 1 elements in a 1D array 
myCount = lambda ar: [sum(val for _ in group) for val, group in groupby(ar) if val==1]

# count consecutive 1 elements in a 1D array
def count_runs_ones(binary_array):
    """Return lengths of consecutive runs of 1s in a binary array."""
    # Padding with 0s for edge detection
    if not np.any(binary_array):
        return []
    padded = np.pad(binary_array, (1, 1), mode='constant')
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return (ends - starts).tolist()
    
# Helper function to check data availability
def find_avail_data_df(df, tol):
    return df.notna().mean(axis=1) >= tol

def find_avail_data_array(arr, tol):
    return np.mean(~np.isnan(arr)) >= tol

# Helper function for sinusoidal fit
def sine_func(x, delta, shift):
    return np.mean(x) + delta * np.sin(2 * np.pi * (x - shift) / 365.25)

# References
# McMahon, T. A., Peel, M. C., Lowe, L., Srikanthan, R., and McVicar, T. R.:
# Estimating actual, potential, reference crop and pan evaporation using standard meteorological data: a pragmatic synthesis,
# Hydrol. Earth Syst. Sci., 17, 1331–1363, https://doi.org/10.5194/hess-17-1331-2013, 2013

met_list = ['tair','wind','spfh','swdown','lwdown','psurf']

# -----
# unit conversion
sec2day = 60.0*60.0*24.0
j2mj    = 1e-6   # J to MJ
pa2kpa  = 1e-3   # pa to kPa
kg2m3   = 1e-3   # 1kg of water to m3
m2mm    = 1e+3   # meter to mm

# -----
# Physical constant/parameters
# -----
# molar mass of water [kg/mol] and mass of dry air [kg/mol]
Mw = 18.0153
Md = 28.9647

# wind function parameters
a = 1.313
b = 1.381

# Latent heat of vaporization [MJ kg-1]
latent_vap = 2.45

# Stefan-Boltzman constant [J/s/m2/K4]
SB = 5.670374419e-8

# season definition
SeasonDict = {10:'SON', 11:'SON', 12:'DJF', 1:'DJF', 2:'DJF', 3:'MAM', 4:'MAM', 5:'MAM', 6:'JJA', 7:'JJA', 8:'JJA', 9:'SON'}

# pe computation
def priestley_taylor(sw, lw, u, T, q, p, alpha=1.26, G=0):
    """Compute Priestley-Taylor PET [mm/day]
       input: sw = incoming shortwave radiation [W/m^2]
              lw = incoming longwave radiation [W/m^2]
              u  = wind speed [m/s]
              T  = air temperature [C-degree]
              q  = specific humidity [kg/kg]
              p  = air pressure [kPa]
    """
    Rnet = comp_Rnet(sw, lw, T, q, p)   # [w/m2]
    delta = comp_delta_vp_curve(T)
    psych = comp_psychrometric(p)
    Ept = alpha*(Rnet-G)*(delta/(delta+psych))
    return Ept

def Penman(sw, lw, u, T, q, p):
    """Compute penman open-watr Evaporation mm/day
       input: sw = incoming shortwave radiation [W/m^2]
              lw = incoming longwave radiation [W/m^2]
              u  = wind speed [m/s]
              T  = air temperature [C-degree]
              q  = specific humidity [kg/kg]
              p  = air pressure [kPa]
    """

    Rnet = comp_Rnet(sw, lw, T, q, p)   # [w/m2]
    delta = comp_delta_vp_curve(T)
    psych = comp_psychrometric(p)
    Ea = comp_Ea(u, T, q, p)

    Epen1 = delta/(delta + psych)*Rnet/(latent_vap/j2mj) * kg2m3 * m2mm * sec2day  #[mm/day]
    Epen = Epen1 + psych/(delta + psych)*Ea   # mm/day

    return Epen

def comp_Rnet(sw, lw, T, q, p, albedo=0.08):
    """ Compute net Radiation [w/m^2]
        input: sw = incoming shortwave radiation [W/m^2]
               lw = incoming longwave radiation [W/m^2]
               T  = air temperature [C-degree]
               q  = specific humidity [kg/kg]
               p  = air pressure [kPa]
               albedo = water albedo
    """
    Twb = comp_wb_T(T, q, p)
    lw_out = SB*(T + 273.15)**4 + 4.0*SB*(T + 273.15)**3*(Twb-T)
    Rnet = (1.0 - albedo)*sw + lw - lw_out
    return Rnet

def comp_wb_T(T, q, p):
   """ Compute web-bulb temperature (simpler version) [C-degree]
        input: T = air temperature [C-degree]
               q = specific humidity [kg/kg]
               p = air pressure [kPa]
   """
   va = comp_va(q, p)
   Td = (116.9+237.3*np.log(va))/(16.78-np.log(va))
   Twb = (0.00066*100.0*T + 4098*va*Td/(Td + 237.3)**2)/(0.00066*100.0+4098*va/(Td + 237.3)**2)
   return Twb

def comp_delta_vp_curve(T):
    """ Compute Slope of the saturation vapour pressure curve [kPa/C]
        input:  T = air temperature [C-degree]
    """
    delta = 4098*(0.6108*np.exp(17.27*T/(T + 237.3)))/(T + 237.3)**2
    return delta

def comp_Ea(u, T, q, p):
    """ Compute aerodynamic equation [mm/day]
        input: u = wind speed [m/s]
               T = air temperature [C-degree]
               q = specific humidity [kg/kg]
               p = air pressure [kPa]
    """
    uadj = wind_adjust(u,10, 2)
    u_function = a+b*uadj
    va_sat = comp_va_sat(T)
    va     = comp_va(q, p)
    return u_function*(va_sat-va)

def comp_psychrometric(p):
    """ Compute Psychrometric constant
        input:  p = air pressure [kPa]
    """
    return 0.00163*p/latent_vap

def comp_va(q, p):
    """ Compute vapor pressure [kPa]
        input: q = specific humidity [kg/kg]
               p = air pressure [kPa]
    """
    # mixing ratio [kg/kg] from specific humidity [kg/kg]
    w = q/(1.0-q)
    eps = Mw/Md
    return w*p/(eps+w)

def comp_va_sat(T):
    """ Compute saturation vapor pressure [kPa]
        input: T = air temperature [C-degree]
    """
    #vp_sat = np.exp(34.494-4924.99/(T+237.1))/(T+105.0)**1.57
    #vp_sat = vp_sat * pa2kpa
    vp_sat = 0.6108*np.exp(17.27*T/(T+237.3))
    return vp_sat

def wind_adjust(u, z0, z1, roughness=0.001):
    """ adjust wind speed [m/s] at z0 to heigh at z1
        input: u = wind speed at z0 [m/s]
               z0 = measurement height [m]
               z1 = desired height [m]
               roughness roughness height [m]
    """
    return u*np.log(z1/roughness)/np.log(z0/roughness)

def model_tair(d, amp, phase, offset):
    return offset + amp * np.sin(2*np.pi/365*(d + phase))

def model_prec(d, amp, phase, offset):
    return offset *(1 + amp* np.sin(2*np.pi/365*(d + phase)))
    
def seasonality_index(dr_tair, dr_prec):
    '''
     Function to compute precipitation seasonality
     the combination of aridity, fraction of precipitation falling as snow and precipitation
     seasonality was proposed by Berghuijs et al., 2014, WRR, doi:10.1002/2014WR015692

     return
    '''
    epsilon = 0.00001
    dr_hru = dr_prec['hru']
    
    # long-term mean of year of day values dim = [dayofyear, hru]
    dr_t_day_season = dr_tair.groupby("time.dayofyear").mean()
    dr_p_day_season = dr_prec.groupby("time.dayofyear").mean().rolling(dayofyear=30, center=True, min_periods=1).mean()
    # mean values: dim=[hru]
    dr_t_mean = dr_t_day_season.mean(dim='dayofyear')
    dr_p_mean = dr_p_day_season.mean(dim='dayofyear')
    # amplitude: dim=[hru]
    dr_t_amplitude = dr_t_day_season.max(dim='dayofyear')-dr_t_day_season.min(dim='dayofyear')
    dr_p_amplitude = dr_p_day_season.max(dim='dayofyear')-dr_p_day_season.min(dim='dayofyear')
    
    # Fit temperature and precipitation curves
    # first guess of parameters for precipitation curves: dim = [hru]
    mask_all_nan = dr_p_day_season.isnull().all(dim="dayofyear")
    dr_p_day_season_mod = dr_p_day_season.where(~mask_all_nan, other=-1)    
    s_p_first_guess = (90 - dr_p_day_season_mod.argmax(dim='dayofyear')*30)/360

    dayofyear=dr_t_day_season.dayofyear.values

    ds_p_index = xr.Dataset(data_vars=dict(
        p_seasonality = (["hru"], np.full(len(dr_hru['hru']), np.nan, dtype='float32')),
        snow_frac = (["hru"], np.full(len(dr_hru['hru']), np.nan, dtype='float32')),),
        coords=dict(hru=dr_hru,),
                           )
    ix_missing = np.where(np.all(np.isnan(dr_p_day_season.values), axis=0))[0] # get hru index where value is nan at all the time step
    
    for ix, hru in enumerate(dr_hru.values):
        if ix in ix_missing:
            continue
        p0=[dr_t_amplitude.sel(hru=hru).values/2, 30, dr_t_mean.sel(hru=hru).values]
        p1=[0.5, s_p_first_guess.sel(hru=hru).values, dr_p_mean.sel(hru=hru).values]
        fit_t, _ = curve_fit(model_tair, dayofyear, dr_t_day_season.sel(hru=hru).values, 
                      p0=p0,
                      #bounds=((-np.inf, -np.inf, dr_t_mean.sel(hru=hru).values-epsilon), (np.inf, np.inf, dr_t_mean.sel(hru=hru).values+epsilon)),
                      ftol=0.05, xtol=0.05,) #maxfev=500
        fit_p, _ = curve_fit(model_prec, dayofyear, dr_p_day_season.sel(hru=hru).values, 
                          p0=p1,
                          bounds=((-1, -np.inf, dr_p_mean.sel(hru=hru).values-epsilon), (1, np.inf, dr_p_mean.sel(hru=hru).values+epsilon)),
                          method='trf', ftol=0.01, xtol=0.01,) 
        
        delta_t, s_t = fit_t[0:2]
        delta_p, s_p = fit_p[0:2]
        delta_p_star =  delta_p * np.sign(delta_t) * np.cos(2*np.pi*(s_p - s_t) /365)
        ds_p_index['p_seasonality'].loc[hru] = delta_p_star

        t_0 = 1 # snow-rain partioning temperature
        t_star_bar = (dr_t_mean.sel(hru=hru).values - t_0) / abs(delta_t)
        if t_star_bar > 1:
            f_s = 0
        elif t_star_bar < -1:
            f_s = 1
        else:
            f_s = 0.5 - np.arcsin(t_star_bar)/np.pi - delta_p_star/np.pi*np.sqrt(1 - t_star_bar ** 2)
        
        # fraction of precipitation falling as snow - using daily temp and precip values
        if np.any((dr_tair.sel(hru=hru).values <= 0) & (dr_prec.sel(hru=hru).values > 0)): 
            f_s_daily = np.sum(dr_prec.sel(hru=hru).values[dr_tair.sel(hru=hru).values <= 0]) / np.sum(dr_prec.sel(hru=hru).values) 
        else:
            f_s_daily = 0
        ds_p_index['snow_frac'].loc[hru] = f_s_daily
        
    return ds_p_index


def high_p_freq_dur(dr: xr.DataArray, p_thresh_mult=5, dayofyear='wateryear') -> xr.Dataset:
    """
    Calculates frequency and duration of high-precip events (> p_thresh_mult * mean) and their seasonal timing.
    freq_high_p: frequency of high-flow days (> p_thresh_mult times the mean daily flow) day/yr
    mean_high_p_dur: average duration of high-precipitation events over yr (number of consecutive days > 5 times the mean daily flow) day
    """

    if dayofyear == 'wateryear':
        smon, sday, emon, eday, yr_adj = 10, 1, 9, 30, 1
    elif dayofyear == 'calendar':
        smon, sday, emon, eday, yr_adj = 1, 1, 12, 31, 0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    hru_list = dr['hru'].values
    n_hrus = len(hru_list)
    years = np.unique(dr.time.dt.year.values)
    if dayofyear == 'wateryear':
        years = years[:-1]
        if len(years) == 0:
            raise Exception('Insufficient years for wateryear analysis.')

    # Pre-compute threshold per HRU
    q_thresh = dr.mean(dim='time').values * p_thresh_mult

    # Initialize output arrays
    freq = np.full((len(years), n_hrus), np.nan, dtype='float32')
    dur = np.full_like(freq, np.nan)
    timing = np.full((len(years), n_hrus), 'None', dtype=object)

    for yi, yr in enumerate(years):
        time_slice = slice(f'{yr}-{smon:02}-{sday:02}', f'{yr + yr_adj}-{emon:02}-{eday:02}')
        dr_yr = dr.sel(time=time_slice)
        q_array = dr_yr.values
        time_vals = dr_yr['time'].values
        months = pd.to_datetime(time_vals).month
        seasons = np.array([SeasonDict[m] for m in months])

        # Apply per HRU
        for h in range(n_hrus):
            binary = (q_array[:, h] > q_thresh[h]).astype(np.int8)
            runs = count_runs_ones(binary)

            freq[yi, h] = len(runs)
            dur[yi, h] = np.mean(runs) if runs else 0

            if np.any(binary):
                season_counts = pd.Series(binary).groupby(seasons).sum()
                if not season_counts.empty:
                    timing[yi, h] = season_counts.idxmax()

    return xr.Dataset(
        data_vars=dict(
            high_prec_freq=(["year", "hru"], freq),
            high_prec_dur=(["year", "hru"], dur),
            high_prec_timing=(["year", "hru"], timing),
        ),
        coords=dict(
            year=years,
            hru=hru_list,
        ),
    )

def low_p_freq_dur(dr: xr.DataArray, day_p_thresh_mm=1, dayofyear='wateryear', how2count_freq='day') -> xr.Dataset:
    """Efficiently compute frequency, duration, and seasonal timing of low-precip events."""
    
    if dayofyear == 'wateryear':
        smon, sday, emon, eday, yr_adj = 10, 1, 9, 30, 1
    elif dayofyear == 'calendar':
        smon, sday, emon, eday, yr_adj = 1, 1, 12, 31, 0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    hru_list = dr['hru'].values
    n_hrus = len(hru_list)
    years = np.unique(dr.time.dt.year.values)
    if dayofyear == 'wateryear':
        years = years[:-1]
        if len(years) == 0:
            raise Exception('Insufficient years for wateryear analysis.')

    # Prepare output arrays
    freq = np.full((len(years), n_hrus), np.nan, dtype='float32')
    dur = np.full_like(freq, np.nan)
    timing = np.full((len(years), n_hrus), 'None', dtype=object)

    for yi, yr in enumerate(years):
        time_slice = slice(f'{yr}-{smon:02}-{sday:02}', f'{yr + yr_adj}-{emon:02}-{eday:02}')
        year_data = dr.sel(time=time_slice)
        p_array = year_data.values  # shape: (days, hru)
        time_vals = pd.to_datetime(year_data['time'].values)
        seasons = np.array([SeasonDict[m] for m in time_vals.month])

        for h in range(n_hrus):
            binary = (p_array[:, h] < day_p_thresh_mm).astype(np.int8)
            runs = count_runs_ones(binary)

            freq_val = 0
            if how2count_freq == 'event':
                freq_val = len(runs)
            elif how2count_freq == 'day':
                freq_val = np.count_nonzero(binary)
            else:
                raise ValueError('Invalid "how2count_freq": should be "event" or "day"')

            freq[yi, h] = freq_val
            dur[yi, h] = np.mean(runs) if runs else 0

            if np.any(binary):
                season_counts = pd.Series(binary).groupby(seasons).sum()
                if not season_counts.empty:
                    timing[yi, h] = season_counts.idxmax()

    return xr.Dataset(
        data_vars=dict(
            low_prec_freq=(["year", "hru"], freq),
            low_prec_dur=(["year", "hru"], dur),
            low_prec_timing=(["year", "hru"], timing),
        ),
        coords=dict(
            year=years,
            hru=hru_list,
        ),
    )

#--- dask aware function 

def process_high_series(series, months, thresh):
    """Process one HRU time series to compute high-p freq/dur/timing."""
    binary = (series > thresh).astype(np.int8)
    runs = count_runs_ones(binary)
    freq_val = len(runs)
    dur_val = float(np.mean(runs)) if runs else 0.0

    if np.any(binary):
        seasons = np.array([SeasonDict[m] for m in months])
        season_counts = pd.Series(binary).groupby(seasons).sum()
        timing_val = season_counts.idxmax() if not season_counts.empty else 'None'
    else:
        timing_val = 'None'

    return np.array([freq_val, dur_val], dtype=np.float32), timing_val

def low_p_freq_dur_dask(dr: xr.DataArray, p_thresh_mult=5, dayofyear='wateryear') -> xr.Dataset:
    """Dask-aware calculation of high-precip event frequency, duration, and timing."""

    if dayofyear == 'wateryear':
        smon, sday, emon, eday, yr_adj = 10, 1, 9, 30, 1
    elif dayofyear == 'calendar':
        smon, sday, emon, eday, yr_adj = 1, 1, 12, 31, 0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)
    if dayofyear == 'wateryear':
        years = years[:-1]
        if len(years) == 0:
            raise Exception('Insufficient years for wateryear analysis.')

    hru_list = dr['hru'].values
    q_thresh = dr.mean(dim='time') * p_thresh_mult

    # Storage for annual results
    freq_dur_list = []
    timing_list = []

    for yr in years:
        time_slice = slice(f"{yr}-{smon:02}-{sday:02}", f"{yr + yr_adj}-{emon:02}-{eday:02}")
        dr_yr = dr.sel(time=time_slice)
        months = dr_yr['time'].dt.month
        months_array = xr.DataArray(
            np.broadcast_to(months.values[:, None], dr_yr.shape),
            dims=("time", "hru"),
            coords={"time": dr_yr.time, "hru": dr_yr.hru}
        )

        # Apply to each HRU time series
        freq_dur, timing = xr.apply_ufunc(
            lambda series, m, thresh: process_high_series(series, m, thresh),
            dr_yr,
            months_array,
            q_thresh,
            input_core_dims=[["time"], ["time"], []],
            output_core_dims=[["metric"], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32, object],
        )

        freq_dur = freq_dur.expand_dims(year=[yr])
        timing = timing.expand_dims(year=[yr])

        freq_dur_list.append(freq_dur)
        timing_list.append(timing)

    # Combine results across years
    freq_dur_all = xr.concat(freq_dur_list, dim="year")
    timing_all = xr.concat(timing_list, dim="year")

    return xr.Dataset(
        data_vars=dict(
            high_prec_freq=(["year", "hru"], freq_dur_all.sel(metric=0)),
            high_prec_dur=(["year", "hru"], freq_dur_all.sel(metric=1)),
            high_prec_timing=(["year", "hru"], timing_all),
        ),
        coords=dict(
            year=years,
            hru=hru_list,
        ),
    )


def process_one_series(binary, months, how2count_freq):
    """Helper to compute freq, dur, and timing from binary and months."""
    binary = binary.astype(np.int8)
    runs = count_runs_ones(binary)
    freq = len(runs) if how2count_freq == 'event' else np.count_nonzero(binary)
    dur = float(np.mean(runs)) if runs else 0.0

    if np.any(binary):
        seasons = np.array([SeasonDict[m] for m in months])
        season_counts = pd.Series(binary).groupby(seasons).sum()
        timing = season_counts.idxmax() if not season_counts.empty else 'None'
    else:
        timing = 'None'

    return np.array([freq, dur]), timing

def low_p_freq_dur_dask(dr: xr.DataArray, day_p_thresh_mm=1, dayofyear='wateryear', how2count_freq='day') -> xr.Dataset:
    """Dask-aware computation of low precip frequency, duration, and seasonal timing."""
    
    if dayofyear == 'wateryear':
        smon, sday, emon, eday, yr_adj = 10, 1, 9, 30, 1
    elif dayofyear == 'calendar':
        smon, sday, emon, eday, yr_adj = 1, 1, 12, 31, 0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)
    if dayofyear == 'wateryear':
        years = years[:-1]
        if len(years) == 0:
            raise Exception('Insufficient years for wateryear analysis.')

    hru_list = dr['hru'].values
    n_hrus = len(hru_list)

    results_freq_dur = []
    results_timing = []

    for yr in years:
        time_slice = slice(f'{yr}-{smon:02}-{sday:02}', f'{yr + yr_adj}-{emon:02}-{eday:02}')
        dr_yr = dr.sel(time=time_slice)
        binary = (dr_yr < day_p_thresh_mm)
        binary = binary.chunk({'time': -1})  # ensure time is not chunked

        months = dr_yr['time'].dt.month.values  # eagerly evaluated since small
        months_bcast = np.broadcast_to(months[:, None], binary.shape)

        # Apply the function across time axis for each HRU
        freq_dur, timing = xr.apply_ufunc(
            lambda b, m: process_one_series(b, m, how2count_freq),
            binary,
            months_bcast,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[["metric"], []],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[np.float32, object],
        ).compute()  # compute here to collect results year by year

        results_freq_dur.append(freq_dur)
        results_timing.append(timing)

    # Stack results
    results_freq_dur = np.stack(results_freq_dur)  # shape: (years, hru, metric)
    results_timing = np.stack(results_timing)      # shape: (years, hru)

    return xr.Dataset(
        data_vars=dict(
            low_prec_freq=(["year", "hru"], results_freq_dur[:, :, 0]),
            low_prec_dur=(["year", "hru"], results_freq_dur[:, :, 1]),
            low_prec_timing=(["year", "hru"], results_timing),
        ),
        coords=dict(
            year=years,
            hru=hru_list,
        ),
    )