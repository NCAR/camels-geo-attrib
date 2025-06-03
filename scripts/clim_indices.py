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
def count_runs_ones(binary_array):
    """Return lengths of consecutive runs of 1s in a binary array."""
    # Padding with 0s for edge detection
    padded = np.pad(binary_array, (1, 1), mode='constant')
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return ends - starts

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
    freq_high_p: frequency of high-flow days (> p_thresh_mult times the mean daily flow) day/yr
    mean_high_p_dur: average duration of high-precipitation events over yr (number of consecutive days > 5 times the mean daily flow) day
    """
    years = np.unique(dr.time.dt.year.values)
    if dayofyear=='wateryear':
        years = years[:-1]
        if len(years)==0:
            raise Exception('Invalid argument for "dayofyear"')
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')
        
    hru_list = dr['hru'].values
    n_hrus = len(hru_list)
    
    ds_high_p = xr.Dataset(data_vars=dict(
                    high_prec_dur =(["year", "hru"], np.full((len(years),n_hrus), np.nan, dtype='float32')),
                    high_prec_freq =(["year", "hru"], np.full((len(years),n_hrus), np.nan, dtype='float32')),
                    high_prec_timing =(["year", "hru"], np.full((len(years),n_hrus), 'None', dtype=np.object_)),
                    ),
                    coords=dict(year=years,
                                hru=hru_list,),
                    )

    t_axis = dr.dims.index('time')
    q_thresh=np.mean(dr.values, axis=t_axis)*p_thresh_mult

    for yi, yr in enumerate(years):
        time_slice=slice(f'{yr}-{smon:02}-{sday:02}',f'{yr+yr_adj}-{emon:02}-{eday:02}')
        year_data = dr.sel(time=time_slice)
        q_array = year_data.values
        times = year_data['time'].values
        months = pd.to_datetime(times).month
        seasons = np.array([SeasonDict[m] for m in months])
        
        for h in range(n_hrus):
            binary = (q_array[:,h] > q_thresh[h]).astype(np.int8)
            runs = count_runs_ones(binary)
            
            if len(runs) == 0:
                ds_high_p['high_prec_dur'][yi, h] = 0
                ds_high_p['high_prec_freq'][yi, h] = 0
            else:
                ds_high_p['high_prec_dur'][yi, h] = np.mean(runs)
                ds_high_p['high_prec_freq'][yi, h] = len(runs) # number of events -> len(runs)

                season_counts = pd.Series(binary).groupby(seasons).sum()
                if not season_counts.empty:
                    ds_high_p['high_prec_timing'][yi, h] = season_counts.idxmax()

    return ds_high_p


def low_p_freq_dur(dr: xr.DataArray, day_p_thresh_mm=1, dayofyear='wateryear', how2count_freq='day') -> xr.Dataset:
    # : frequency of low-precipitation days (< 1 mm/day) day/yr
    # : average duration of low-precipitation events (number of consecutive days < 0.2 times the mean daily flow) day

    years = np.unique(dr.time.dt.year.values)
    if dayofyear=='wateryear':
        years = years[:-1]
        if len(years)==0:
            raise Exception('Invalid argument for "dayofyear"')
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    hru_list = dr['hru'].values
    n_hrus = len(hru_list)

    ds_low_p = xr.Dataset(
        data_vars=dict(
            low_prec_dur=(["year", "hru"], np.full((len(years), n_hrus), np.nan, dtype='float32')),
            low_prec_freq=(["year", "hru"], np.full((len(years), n_hrus), np.nan, dtype='float32')),
            low_prec_timing=(["year", "hru"], np.full((len(years), n_hrus), 'None', dtype=object)),
        ),
        coords=dict(year=years, hru=hru_list),
    )

    for yi, yr in enumerate(years):
        time_slice = slice(f'{yr}-{smon:02}-{sday:02}', f'{yr + yr_adj}-{emon:02}-{eday:02}')
        year_data = dr.sel(time=time_slice)
        p_array = year_data.values
        times = year_data['time'].values

        months = pd.to_datetime(times).month
        seasons = np.array([SeasonDict[m] for m in months])

        for h in range(n_hrus):
            binary = (p_array[:, h] < day_p_thresh_mm).astype(np.int8)
            runs = count_runs_ones(binary)

            if len(runs) == 0:
                ds_low_p['low_prec_freq'][yi, h] = 0
                ds_low_p['low_prec_dur'][yi, h] = 0
                ds_low_p['low_prec_timing'][yi, h] = 'None'
            else:
                ds_low_p['low_prec_dur'][yi, h] = np.mean(runs)
                if how2count_freq == 'event':
                    ds_low_p['low_prec_freq'][yi, h] = len(runs)
                elif how2count_freq == 'day':
                    ds_low_p['low_prec_freq'][yi, h] = np.count_nonzero(binary)
                else:
                    raise ValueError('Invalid "how2count_freq": should be "event" or "day"')

                # Seasonal timing (mode of seasonal counts)
                season_counts = pd.Series(binary).groupby(seasons).sum()
                if not season_counts.empty:
                    ds_low_p['low_prec_timing'][yi, h] = season_counts.idxmax()

    return ds_low_p

#--- old scripts
# count consecutive 1 elements in a 1D array 
myCount = lambda ar: [sum(val for _ in group) for val, group in groupby(ar) if val==1]

def high_p_freq_dur_slow(dr: xr.DataArray, p_thresh_mult=5, dayofyear='wateryear') -> xr.Dataset:
    """
    freq_high_p: frequency of high-flow days (> p_thresh_mult times the mean daily flow) day/yr
    mean_high_p_dur: average duration of high-precipitation events over yr (number of consecutive days > 5 times the mean daily flow) day
    """
    years = np.unique(dr.time.dt.year.values)
    if dayofyear=='wateryear':
        years = years[:-1]
        if len(years)==0:
            raise Exception('Invalid argument for "dayofyear"')
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    ds_high_p = xr.Dataset(data_vars=dict(
                    high_prec_dur =(["year", "hru"], np.full((len(years),len(dr['hru'])), np.nan, dtype='float32')),
                    high_prec_freq =(["year", "hru"], np.full((len(years),len(dr['hru'])), np.nan, dtype='float32')),
                    high_prec_timing =(["year", "hru"], np.full((len(years),len(dr['hru'])), 'None', dtype=np.object_)),
                    ),
                    coords=dict(year=years,
                                hru=dr['hru'],),
                    )

    t_axis = dr.dims.index('time')
    q_thresh=np.mean(dr.values, axis=t_axis)*p_thresh_mult

    for yr in years:
        time_slice=slice(f'{yr}-{smon}-{sday}',f'{yr+yr_adj}-{emon}-{eday}')
        datetime1 = dr['time'].sel(time=time_slice)
        
        q_array = dr.sel(time=time_slice).values # find annual max flow
        for sidx, hru in enumerate(dr['hru'].values):
            binary_array = np.where(q_array[:,sidx] > q_thresh[sidx], 1, 0)
            count_dups = myCount(binary_array)
            if not count_dups:
                ds_high_p['high_prec_dur'].loc[yr, hru] = 0
                ds_high_p['high_prec_freq'].loc[yr, hru] = 0
            else:
                ds_high_p['high_prec_dur'].loc[yr, hru] = np.mean(count_dups)
                ds_high_p['high_prec_freq'].loc[yr, hru] = len(count_dups) # number of events -> len(count_dups)
                df = pd.DataFrame(index=datetime1, data=binary_array)
                max_season = df.groupby(lambda x: SeasonDict.get(x.month)).sum().idxmax().values
                ds_high_p['high_prec_timing'].loc[yr, hru] = max_season[0]

    return ds_high_p


def low_p_freq_dur_slow(dr: xr.DataArray, day_p_thresh_mm=1, dayofyear='wateryear', how2count_freq='day') -> xr.Dataset:
    # : frequency of low-precipitation days (< 1 mm/day) day/yr
    # : average duration of low-precipitation events (number of consecutive days < 0.2 times the mean daily flow) day
    years = np.unique(dr.time.dt.year.values)
    if dayofyear=='wateryear':
        years = years[:-1]
        if len(years)==0:
            raise Exception('Invalid argument for "dayofyear"')
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    ds_low_p = xr.Dataset(data_vars=dict(
                    low_prec_dur =(["year", "hru"], np.full((len(years),len(dr['hru'])), np.nan, dtype='float32')),
                    low_prec_freq =(["year", "hru"], np.full((len(years),len(dr['hru'])), np.nan, dtype='float32')),
                    low_prec_timing =(["year", "hru"], np.full((len(years),len(dr['hru'])), 'None', dtype=np.object_)),
                    ),
                    coords=dict(year=years,
                                hru=dr['hru'],),
                    )

    t_axis = dr.dims.index('time')

    for yr in years:
        time_slice=slice(f'{yr}-{smon}-{sday}',f'{yr+yr_adj}-{emon}-{eday}')
        datetime1 = dr['time'].sel(time=time_slice)
        
        p_array = dr.sel(time=time_slice).values # precipitation for one year
        for sidx, hru in enumerate(dr['hru'].values):
            binary_array = np.where(p_array[:,sidx] < day_p_thresh_mm, 1, 0)
            count_dups = myCount(binary_array)
            if not count_dups:
                ds_low_p['low_prec_freq'].loc[yr, hru] = 0
                ds_low_p['low_prec_dur'].loc[yr, hru] = 0
                ds_low_p['low_prec_timing'].loc[yr, hru] = 'None'
            else:
                ds_low_p['low_prec_dur'].loc[yr, hru] = np.mean(count_dups)
                if how2count_freq=='event':
                    ds_low_p['low_prec_freq'].loc[yr, hru] = len(count_dups) # number of events -> len(count_dups)
                elif how2count_freq=='day':
                    ds_low_p['low_prec_freq'].loc[yr, hru] = np.count_nonzero(binary_array) # number of days -> len(count_dups)
                else:
                    raise ValueError('Dang, Invalid argument for "how2count_freq": should be "event" or "day"')
                
                df = pd.DataFrame(index=datetime1, data=binary_array)
                max_season = df.groupby(lambda x: SeasonDict.get(x.month)).sum().idxmax().values
                ds_low_p['low_prec_timing'].loc[yr, hru] = max_season[0]
                
    return ds_low_p