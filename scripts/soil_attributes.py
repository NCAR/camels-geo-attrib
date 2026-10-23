#!/usr/bin/env python

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# some unit conversions
# --time
hr2sec  = 3600. # hour -> sec
# --length
inch2m = 0.0254  # inch -> meter
cm2m   = 0.01  # centi-meter -> meter
# --pressure
cmH2O2kPa = 0.0980665  #
kPa2mH2O = 0.10199773339984
# -- density
kgcm2gcm = 0.001 #kg/m3 -> g/cm3

def retention_slope_ptf1(sand: xr.DataArray, clay: xr.DataArray):
    """
    Compute slope of retention curve in log scale using a transfer function from:

    Cosby, B. J., G. M. Hornberger, R. B. Clapp, and T. R. Ginn (1984),
    A Statistical Exploration of the Relationships of Soil Moisture Characteristics to the Physical Properties of Soils, Water Resour. Res., 20(6), 682–690,
    doi:10.1029/WR020i006p00682.

    Note: soil texture needs to be in percent

    Arguments
    ----------
    sand: xr.DataArray
         sand fraction. Unit is either fraction, frac, or percent
    silt: xr.DataArray
         silt fraction. Unit is either fraction, frac, or percent

    Returns
    -------
    xr.DataArray:
         slope of retention curve in log scale [-]
    """

    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if clay.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if sand.attrs['units'] == 'frac' or sand.attrs['units'] == 'fraction':
        sand = sand * 100.0
    if clay.attrs['units'] == 'frac' or clay.attrs['units'] == 'fraction':
        clay = clay * 100.0

    a = 3.1 + 0.157*clay- 0.003*sand
    return a

def matric_potential_ptf1(sand: xr.DataArray, silt: xr.DataArray):
    """
    Compute matric potential [kPa] using a transfer function from:

    Cosby, B. J., G. M. Hornberger, R. B. Clapp, and T. R. Ginn (1984),
    A Statistical Exploration of the Relationships of Soil Moisture Characteristics to the Physical Properties of Soils, Water Resour. Res., 20(6), 682–690,
    doi:10.1029/WR020i006p00682.

    Note: soil texture needs to be in percent

    Arguments
    ----------
    sand: xr.DataArray
         sand fraction. Unit is either fraction, or percent, or g/kg
    silt: xr.DataArray
         silt fraction. Unit is either fraction, or percent, or g/kg

    Returns
    -------
    xr.DataArray:
         saturation matric potential [kPa]
    """

    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if silt.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if sand.attrs['units'] == 'frac' or sand.attrs['units'] == 'fraction':
        sand = sand * 100.0
    if silt.attrs['units'] == 'frac' or silt.attrs['units'] == 'fraction':
        silt = silt * 100.0

    a = -1* (10.0**(1.54- 0.0095* sand+ 0.0063* silt))*cmH2O2kPa # cosby equation give cm-H2O
    return a

def porosity_ptf1(usda_soil_class: xr.DataArray, soilCategory='STATS-RUC') -> xr.DataArray:
    """
    Obtain porosity based on  USDA soil classification. Use noah-mp soilparm table. See Table 3 in:

    Campbell, P. C., Bash, J. O., & Spero, T. L. (2019).
    Updates to the Noah land surface model in WRF-CMAQ to improve simulated meteorology, air quality, and deposition. Journal of Advances in Modeling Earth Systems, 11, 231–256.
    https://doi.org/10.1029/2018MS001422

    Arguments
    ----------
    usda_soil_class: xr.DataArray
         soil class ID. 1-12

    Returns
    -------
    xr.DataArray:
         permanent wilting point [frac]
    """

    # Classification logic
    #1: SAND, 2: LOAMY SAND, 3: SANDY LOAM,
    #4: SILT LOAM, 5: SILT, 6: LOAM, 7: SANDY CLAY LOAM, 8: SILTY CLAY LOAM,
    #9: CLAY LOAM, 10: SANDY CLAY, 11: SILTY CLAY, 12:CLAY
    coords = {}
    for coord_name in usda_soil_class.coords:
        coords[coord_name] = (list(usda_soil_class.coords[coord_name].dims), usda_soil_class[coord_name].values)
    a = xr.DataArray(
        data=np.ones(usda_soil_class.shape, dtype=np.float32)*-999.,
        dims=list(usda_soil_class.dims),
        coords=coords,
        attrs={'long_name':'porosity', 'unit':'-'}
    )
    if soilCategory=='STAS':
       # Classification logic
        a = xr.where(usda_soil_class==1, 0.339, a)  # 1 'SAND'
        a = xr.where(usda_soil_class==2, 0.421, a) # 2 'LOAMY SAND'
        a = xr.where(usda_soil_class==3, 0.434, a) # 3 'SANDY LOAM'
        a = xr.where(usda_soil_class==4, 0.476, a) # 4 'SILT LOAM'
        a = xr.where(usda_soil_class==5, 0.476, a) # 5 'SILT'
        a = xr.where(usda_soil_class==6, 0.439, a) # 6 'LOAM'
        a = xr.where(usda_soil_class==7, 0.404, a) # 7 'SANDY CLAY LOAM'
        a = xr.where(usda_soil_class==8, 0.464, a) # 8 'SILTY CLAY LOAM'
        a = xr.where(usda_soil_class==9, 0.465, a) # 9 'CLAY LOAM'
        a = xr.where(usda_soil_class==10, 0.406, a) # 10 'SANDY CLAY'
        a = xr.where(usda_soil_class==11, 0.468, a) # 11 'SILTY CLAY'
        a = xr.where(usda_soil_class==12, 0.468, a) # 12 'CLAY'
    elif soilCategory=='STAS-RUC':
        # Classification logic
        a = xr.where(usda_soil_class==1, 0.395, a)  # 1 'SAND'
        a = xr.where(usda_soil_class==2, 0.410, a) # 2 'LOAMY SAND'
        a = xr.where(usda_soil_class==3, 0.435, a) # 3 'SANDY LOAM'
        a = xr.where(usda_soil_class==4, 0.485, a) # 4 'SILT LOAM'
        a = xr.where(usda_soil_class==5, 0.485, a) # 5 'SILT'
        a = xr.where(usda_soil_class==6, 0.451, a) # 6 'LOAM'
        a = xr.where(usda_soil_class==7, 0.420, a) # 7 'SANDY CLAY LOAM'
        a = xr.where(usda_soil_class==8, 0.477, a) # 8 'SILTY CLAY LOAM'
        a = xr.where(usda_soil_class==9, 0.476, a) # 9 'CLAY LOAM'
        a = xr.where(usda_soil_class==10, 0.426, a) # 10 'SANDY CLAY'
        a = xr.where(usda_soil_class==11, 0.492, a) # 11 'SILTY CLAY'
        a = xr.where(usda_soil_class==12, 0.482, a) # 12 'CLAY'

    return a

def porosity_ptf2(clay: xr.DataArray, sand: xr.DataArray, bulk_density: xr.DataArray):
    """
    Compute porosity [fraction] using a transfer function from:

    Zacharias & Wessolek 2007.

    Note:soil texture needs to be in percent

    Arguments
    ----------
    cray: xr.DataArray
         cray fraction. Unit is either fraction, frac, or percent, or g/kg
    sand: xr.DataArray
         sand fraction. Unit is either fraction, frac, or percent
    bulk_density: xr.DataArray
         bulk density. Unit is g/cm3

    Returns
    -------
    xr.DataArray:
         porosity
    """
    if clay.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if bulk_density.attrs['units'] not in ['g/cm^3', 'g/cm3','kg/m^3', 'kg/m3']:
        raise Exception("unit of bulk_density needs to be g/cm^3, g/cm3, kg/m^3 or kg/m3")
    if clay.attrs['units'] == 'frac' or clay.attrs['units'] == 'fraction':
        clay = clay * 100.0
    if sand.attrs['units'] == 'frac' or sand.attrs['units'] == 'fraction':
        sand = sand * 100.0
    if bulk_density.attrs['units'] == 'kg/m^3' or bulk_density.attrs['units'] == 'kg/m3':
        bulk_density = bulk_density *kgcm2gcm

    a = (0.788 + 0.001*clay- 0.263*bulk_density)
    a = a.where(sand<66.5, 0.890-0.001*clay-0.322*bulk_density)

    return a

def porosity_ptf3(sand: xr.DataArray, clay: xr.DataArray):
    """
    Compute porosity [fraction] using a transfer function from:

    Cosby, B. J., G. M. Hornberger, R. B. Clapp, and T. R. Ginn (1984),
    A Statistical Exploration of the Relationships of Soil Moisture Characteristics to the Physical Properties of Soils, Water Resour. Res., 20(6), 682–690,
    doi:10.1029/WR020i006p00682.

    Note: soil texture needs to be in percent

    Arguments
    ----------
    sand: xr.DataArray
         bulk sand. Unit is either fraction, or percent, or g/kg
    cray: xr.DataArray
         cray fraction. Unit is either fraction, frac, or percent

    Returns
    -------
    xr.DataArray:
         porosity [-]
    """

    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if clay.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if sand.attrs['units'] == 'frac' or sand.attrs['units'] == 'fraction':
        sand = sand * 100.0
    if clay.attrs['units'] == 'frac' or clay.attrs['units'] == 'fraction':
        clay = clay * 100.0

    a = (50.5 - 0.142*sand- 0.037*clay)/100.0
    return a

def field_capacity_ptf1(usda_soil_class: xr.DataArray, soilCategory='STATS-RUC') -> xr.DataArray:
    """
    Obtain field capacity based on  USDA soil classification. Use noah-mp soilparm table. see table 3 in:

    Campbell, P. C., Bash, J. O., & Spero, T. L. (2019).
    Updates to the Noah land surface model in WRF-CMAQ to improve simulated meteorology, air quality, and deposition. Journal of Advances in Modeling Earth Systems, 11, 231–256.
    https://doi.org/10.1029/2018MS001422

    Arguments
    ----------
    usda_soil_class: xr.DataArray
         soil class ID. 1-12

    Returns
    -------
    xr.DataArray:
         field capacity [frac]
    """

    # Classification logic
    #1: SAND, 2: LOAMY SAND, 3: SANDY LOAM,
    #4: SILT LOAM, 5: SILT, 6: LOAM, 7: SANDY CLAY LOAM, 8: SILTY CLAY LOAM,
    #9: CLAY LOAM, 10: SANDY CLAY, 11: SILTY CLAY, 12:CLAY
    coords = {}
    for coord_name in usda_soil_class.coords:
        coords[coord_name] = (list(usda_soil_class.coords[coord_name].dims), usda_soil_class[coord_name].values)
    a = xr.DataArray(
        data=np.ones(usda_soil_class.shape, dtype=np.float32)*-999.,
        dims=list(usda_soil_class.dims),
        coords=coords,
        attrs={'long_name':'field capacity', 'unit':'-'}
    )
    if soilCategory=='STAS':
        # Classification logic
        a = xr.where(usda_soil_class==1, 0.236, a)  # 1 'SAND'
        a = xr.where(usda_soil_class==2, 0.383, a) # 2 'LOAMY SAND'
        a = xr.where(usda_soil_class==3, 0.383, a) # 3 'SANDY LOAM'
        a = xr.where(usda_soil_class==4, 0.360, a) # 4 'SILT LOAM'
        a = xr.where(usda_soil_class==5, 0.383, a) # 5 'SILT'
        a = xr.where(usda_soil_class==6, 0.329, a) # 6 'LOAM'
        a = xr.where(usda_soil_class==7, 0.314, a) # 7 'SANDY CLAY LOAM'
        a = xr.where(usda_soil_class==8, 0.387, a) # 8 'SILTY CLAY LOAM'
        a = xr.where(usda_soil_class==9, 0.382, a) # 9 'CLAY LOAM'
        a = xr.where(usda_soil_class==10, 0.338, a) # 10 'SANDY CLAY'
        a = xr.where(usda_soil_class==11, 0.404, a) # 11 'SILTY CLAY'
        a = xr.where(usda_soil_class==12, 0.412, a) # 12 'CLAY'
    elif soilCategory=='STAS-RUC':
        # Classification logic
        a = xr.where(usda_soil_class==1, 0.236, a)  # 1 'SAND'
        a = xr.where(usda_soil_class==2, 0.383, a) # 2 'LOAMY SAND'
        a = xr.where(usda_soil_class==3, 0.383, a) # 3 'SANDY LOAM'
        a = xr.where(usda_soil_class==4, 0.360, a) # 4 'SILT LOAM'
        a = xr.where(usda_soil_class==5, 0.383, a) # 5 'SILT'
        a = xr.where(usda_soil_class==6, 0.329, a) # 6 'LOAM'
        a = xr.where(usda_soil_class==7, 0.314, a) # 7 'SANDY CLAY LOAM'
        a = xr.where(usda_soil_class==8, 0.387, a) # 8 'SILTY CLAY LOAM'
        a = xr.where(usda_soil_class==9, 0.382, a) # 9 'CLAY LOAM'
        a = xr.where(usda_soil_class==10, 0.338, a) # 10 'SANDY CLAY'
        a = xr.where(usda_soil_class==11, 0.404, a) # 11 'SILTY CLAY'
        a = xr.where(usda_soil_class==12, 0.412, a) # 12 'CLAY'
    return a

def field_capacity_ptf2(sand: xr.DataArray, clay: xr.DataArray, occ: xr.DataArray):
    """
    Compute field capacity[frac] using a transfer function from

    Amsili, et al., 2024. soil texture needs to be in percent
    Pedotransfer Functions for Field Capacity, Permanent Wilting Point, and Available Water Capacity Based on Random Forest Models for Routine Soil Health Analysis
    https://doi.org/10.1080/00103624.2024.2336573

    Arguments
    ----------
    sand: xr.DataArray
         sand . Unit is fraction (g/g)
    clay: xr.DataArray
         clay.  Unit is fraction (g/g)
    occ: xr.DataArray
        soil organic carbon.  Unit is percent(%)
    Returns
    -------
    xr.DataArray:
         field capacity
    """
    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if clay.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if sand.attrs['units'] == 'percent':
        sand = sand / 100.0
    if clay.attrs['units'] == 'percent':
        clay = clay / 100.0
    if occ.attrs['units'] == 'percent':
        occ = occ / 100.0

    som=occ/0.58 # soil organic carbon [frac] to soil organic matter [frac]
    a = 0.3477 - 0.313*sand + 0.040*clay + 1.855*som + 0.222*(sand*clay) + 1.405*(sand*som) - 0.202*(clay*som)
    return a

def field_capacity_ptf3(porosity: xr.DataArray, matric_potential: xr.DataArray, retention_slope: xr.DataArray):
    """
    Compute field capacity[frac] using a transfer function from:

    Dingman, Physical hydrology, chapter 6.4

    or

    Cosby, B. J., G. M. Hornberger, R. B. Clapp, and T. R. Ginn (1984),
    A Statistical Exploration of the Relationships of Soil Moisture Characteristics to the Physical Properties of Soils, Water Resour. Res., 20(6), 682–690,
    doi:10.1029/WR020i006p00682.

    Note: matric potential [kPa] and retention curve slope, b have to be computed

    Arguments
    ----------
    porosity: xr.DataArray
         porosity. Unit is fraction
    matric_potential: xr.DataArray
         saturation matric potential. Unit is kPa

    Returns
    -------
    xr.DataArray:
         field capacity
    """
    psi_fc=-33.0 #kPa
    a = porosity*(matric_potential/psi_fc)**(1.0/retention_slope)
    return a

def wilting_point_ptf1(usda_soil_class: xr.DataArray, soilCategory='STATS-RUC') -> xr.DataArray:
    """
    Obtain permanent wilting point based on  USDA soil classification. Use noah-mp soilparm table. See table 3 in

    Campbell, P. C., Bash, J. O., & Spero, T. L. (2019).
    Updates to the Noah land surface model in WRF-CMAQ to improve simulated meteorology, air quality, and deposition. Journal of Advances in Modeling Earth Systems, 11, 231–256.
    https://doi.org/10.1029/2018MS001422

    Arguments
    ----------
    usda_soil_class: xr.DataArray
         soil class ID. 1-12

    Returns
    -------
    xr.DataArray:
         permanent wilting point [frac]
    """

    # Classification logic
    #1: SAND, 2: LOAMY SAND, 3: SANDY LOAM,
    #4: SILT LOAM, 5: SILT, 6: LOAM, 7: SANDY CLAY LOAM, 8: SILTY CLAY LOAM,
    #9: CLAY LOAM, 10: SANDY CLAY, 11: SILTY CLAY, 12:CLAY
    coords = {}
    for coord_name in usda_soil_class.coords:
        coords[coord_name] = (list(usda_soil_class.coords[coord_name].dims), usda_soil_class[coord_name].values)
    a = xr.DataArray(
        data=np.ones(usda_soil_class.shape, dtype=np.float32)*-999.,
        dims=list(usda_soil_class.dims),
        coords=coords,
        attrs={'long_name':'permanent wilting point', 'unit':'-'}
    )
    if soilCategory=='STAS':
        # Classification logic
        a = xr.where(usda_soil_class==1, 0.010, a)  # 1 'SAND'
        a = xr.where(usda_soil_class==2, 0.028, a) # 2 'LOAMY SAND'
        a = xr.where(usda_soil_class==3, 0.047, a) # 3 'SANDY LOAM'
        a = xr.where(usda_soil_class==4, 0.084, a) # 4 'SILT LOAM'
        a = xr.where(usda_soil_class==5, 0.084, a) # 5 'SILT'
        a = xr.where(usda_soil_class==6, 0.066, a) # 6 'LOAM'
        a = xr.where(usda_soil_class==7, 0.067, a) # 7 'SANDY CLAY LOAM'
        a = xr.where(usda_soil_class==8, 0.120, a) # 8 'SILTY CLAY LOAM'
        a = xr.where(usda_soil_class==9, 0.103, a) # 9 'CLAY LOAM'
        a = xr.where(usda_soil_class==10, 0.100, a) # 10 'SANDY CLAY'
        a = xr.where(usda_soil_class==11, 0.126, a) # 11 'SILTY CLAY'
        a = xr.where(usda_soil_class==12, 0.138, a) # 12 'CLAY'
    elif soilCategory=='STAS-RUC':
        # Classification logic
        a = xr.where(usda_soil_class==1, 0.068, a)  # 1 'SAND'
        a = xr.where(usda_soil_class==2, 0.075, a) # 2 'LOAMY SAND'
        a = xr.where(usda_soil_class==3, 0.114, a) # 3 'SANDY LOAM'
        a = xr.where(usda_soil_class==4, 0.179, a) # 4 'SILT LOAM'
        a = xr.where(usda_soil_class==5, 0.179, a) # 5 'SILT'
        a = xr.where(usda_soil_class==6, 0.155, a) # 6 'LOAM'
        a = xr.where(usda_soil_class==7, 0.175, a) # 7 'SANDY CLAY LOAM'
        a = xr.where(usda_soil_class==8, 0.218, a) # 8 'SILTY CLAY LOAM'
        a = xr.where(usda_soil_class==9, 0.250, a) # 9 'CLAY LOAM'
        a = xr.where(usda_soil_class==10, 0.219, a) # 10 'SANDY CLAY'
        a = xr.where(usda_soil_class==11, 0.283, a) # 11 'SILTY CLAY'
        a = xr.where(usda_soil_class==12, 0.286, a) # 12 'CLAY'

    return a

def wilting_point_ptf2(sand: xr.DataArray, clay: xr.DataArray, occ: xr.DataArray):
    """
    Compute permanent wilting point [frac] using a transfer function from

    Amsili, et al., 2024. soil texture needs to be in percent
    Pedotransfer Functions for Field Capacity, Permanent Wilting Point, and Available Water Capacity Based on Random Forest Models for Routine Soil Health Analysis
    https://doi.org/10.1080/00103624.2024.2336573

    Arguments
    ----------
    sand: xr.DataArray
         sand . Unit is fraction (g/g)
    clay: xr.DataArray
         clay.  Unit is fraction (g/g)
    occ: xr.DataArray
        soil organic carbon.  Unit is percent(%)
    Returns
    -------
    xr.DataArray:
         field capacity
    """
    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if clay.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if sand.attrs['units'] == 'percent':
        sand = sand / 100.0
    if clay.attrs['units'] == 'percent':
        clay = clay / 100.0
    if occ.attrs['units'] == 'percent':
        occ = occ / 100.0

    som=occ/0.58 # soil organic carbon [frac] to soil organic matter [frac]
    a = 0.0132 - 0.014*sand + 0.244*clay + 1.158*som
    return a

def wilting_point_ptf3(porosity: xr.DataArray, matric_potential: xr.DataArray, retention_slope: xr.DataArray):
    """
    Compute permanent wilting point [frac] using a transfer function from:

    Dingman, Physical hydrology, chapter 6.4

    or

    Cosby, B. J., G. M. Hornberger, R. B. Clapp, and T. R. Ginn (1984),
    A Statistical Exploration of the Relationships of Soil Moisture Characteristics to the Physical Properties of Soils, Water Resour. Res., 20(6), 682–690,
    doi:10.1029/WR020i006p00682.

    Note: matric potential [kPa] and retention curve slope, b have to be computed

    Arguments
    ----------
    porosity: xr.DataArray
         porosity. Unit is fraction
    matric_potential: xr.DataArray
         saturation matric potential. Unit is kPa

    Returns
    -------
    xr.DataArray:
         permanent wilting point
    """
    psi_wp = -1500.0 #kPa
    a = porosity*(matric_potential/psi_wp)**(1.0/retention_slope)
    return a


def k_sat_ptf1(sand: xr.DataArray, clay: xr.DataArray):
    # cosby et al., 1984 given in inch/hr
    """
    Compute saturation hydraulic conductivity [m/s] using a transfer function from:

    Cosby, B. J., G. M. Hornberger, R. B. Clapp, and T. R. Ginn (1984),
    A Statistical Exploration of the Relationships of Soil Moisture Characteristics to the Physical Properties of Soils, Water Resour. Res., 20(6), 682–690,
    doi:10.1029/WR020i006p00682.

    Note:soil texture needs to be in percent

    Arguments
    ----------
    sand: xr.DataArray
         bulk sand. Unit is either fraction, or percent, or g/kg
    cray: xr.DataArray
         cray fraction. Unit is either fraction, or percent, or g/kg

    Returns
    -------
    xr.DataArray:
         saturation hydraulic conductivity
    """

    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if clay.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if sand.attrs['units'] == 'frac' or sand.attrs['units'] == 'fraction':
        sand = sand * 100.0
    if clay.attrs['units'] == 'frac' or clay.attrs['units'] == 'fraction':
        clay = clay * 100.0

    a = 10.0**(-0.6 + 0.0126* sand - 0.0064* clay)*inch2m/hr2sec
    return a

def vGn_alpha_ptf1(sand: xr.DataArray, clay: xr.DataArray, bulk_density: xr.DataArray, soc: xr.DataArray):
    """
    Compute saturation Van Genuchten alpha parameter [-] using a transfer function from:

    Vereecken et al., 1989: Estimating the soil moisture retention characteristic from texture, bulk density, and carbon content. Soil Sci. 148

    Note:soil texture needs to be in percent.

    Arguments
    ----------
    sand: xr.DataArray
         sand percent. Unit is either fraction, frac, or percent
    clay: xr.DataArray
         clay percent. Unit is either fraction, frac, or percent
    bulk_density: xr.DataArray
         bulk density. Unit is g/cm3
    soc: xr.DataArray
         cray fraction. Unit is either fraction, frac, or percent

    Returns
    -------
    xr.DataArray:
         Van Genuchten alpha parameter
    """
    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if clay.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if soc.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of soil carbon needs to be in percent, frac or fraction")
    if bulk_density.attrs['units'] not in ['g/cm^3', 'g/cm3','kg/m^3', 'kg/m3']:
        raise Exception("unit of bulk_density needs to be g/cm^3, g/cm2, kg/m^3 or kg/m3")

    if sand.attrs['units'] == 'frac' or sand.attrs['units'] == 'fraction':
        sand = sand * 100.0
    if clay.attrs['units'] == 'frac' or clay.attrs['units'] == 'fraction':
        clay = clay * 100.0
    if soc.attrs['units'] == 'frac' or soc.attrs['units'] == 'fraction':
        soc = soc * 100.0
    if bulk_density.attrs['units'] == 'kg/m^3' or bulk_density.attrs['units'] == 'kg/m3':
        bulk_density = bulk_density *kgcm2gcm

    a = -1.0* np.exp(-2.486 + 0.025* sand - 0.351* soc*0.1 -2.617* bulk_density -
                     0.023* clay)/cm2m # this equation use alpha*0.01 for regression alaysis, so have to mutliply 100
    return a

def vGn_n_ptf1(sand: xr.DataArray, clay: xr.DataArray):
    """
    Compute saturation Van Genuchten n parameter [-] using a transfer function from:

    Vereecken et al., 1989: Estimating the soil moisture retention characteristic from texture, bulk density, and carbon content. Soil Sci. 148

    Note:soil texture needs to be in percent.

    Arguments
    ----------
    sand: xr.DataArray
         sand. Unit is either fraction, frac, or percent
    cray: xr.DataArray
         cray fraction. Unit is either fraction, frac, or percent

    Returns
    -------
    xr.DataArray:
         Van Genuchten n parameter
    """

    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if clay.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if sand.attrs['units'] == 'frac' or sand.attrs['units'] == 'fraction':
        sand = sand * 100.0
    if clay.attrs['units'] == 'frac' or clay.attrs['units'] == 'fraction':
        clay = clay * 100.0

    a = np.exp(0.053 + 0.009* sand - 0.013* clay + 0.00015* sand**2)
    a = a.where(a>=1.0, 1.01)
    return a

def max_water_content_ptf1(porosity: xr.DataArray, total_soil_depth: xr.DataArray, layer_thickness: list):
    """
    Compute maximum water content [m]: sum(porosity_i * soil_thickness_i)

    Addor et al., 2017, HESS.

    Arguments
    ----------
    porosity: xr.DataArray [lyr, x, y] or [lyr, hru]
         porosity. Unit is either fraction, frac, or percent
    total_soil_depth: xr.DataArray [x, y]
         total soil thickness. Unit is meter

    Returns
    -------
    xr.DataArray:
         total soil column water content [m]
    """
    thickness_array = np.array(layer_thickness)
    dr_thickness = xr.DataArray(
        data=np.ones_like(porosity.values) * thickness_array[:, np.newaxis, np.newaxis],
        dims=list(porosity.dims),
        coords=porosity.coords,
        attrs={'long_name':'soil thickness'}
    )
    dr_bottom_thickness = total_soil_depth - np.sum(thickness_array)
    dr_bottom_thickness = dr_bottom_thickness.where(~np.isnan(dr_bottom_thickness),0)
    #dr_thickness[-1,:,:] = dr_bottom_thickness
    #a = (porosity[:-1,:,:]* dr_thickness[:-1,:,:]).sum(dim='lyr') + (porosity[-1,:,:]* dr_thickness[-1,:,:])/2 # bottom layer: linearly reduce to zero at the bottom
    a = (porosity* dr_thickness).sum(dim='lyr') + (porosity[-1,:,:]* dr_bottom_thickness)/2 # bottom layer: linearly reduce to zero at the bottom
    return a


# soil classification
def USDA_soil_classification(sand: xr.DataArray, clay: xr.DataArray, silt: xr.DataArray) -> xr.DataArray:
    """
    Compute USDA classification based on sand, clay and slit fraction.

    Note: soil texture should be given in frac

    Arguments
    ----------
    sand: xr.DataArray
         sand fraction. Unit is either fraction, or percent
    clay: xr.DataArray
         clay fraction. Unit is either fraction, or percent
    silt: xr.DataArray
         silt fraction. Unit is either fraction, or percent

    Returns
    -------
    xr.DataArray:
         usda soil class ID
    """

    # Input validation
    if np.min(sand) < 0:
        raise ValueError('sand: values should be positive')

    if np.min(clay) < 0:
        raise ValueError('clay: values should be positive')

    if np.min(silt) < 0:
        raise ValueError('silt: values should be positive')

    if sand.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of sand texture needs to be in percent, frac or fraction")
    if clay.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of clay texture needs to be in percent, frac or fraction")
    if silt.attrs['units'] not in ['percent', 'frac', 'fraction']:
        raise Exception("unit of silt texture needs to be in percent, frac or fraction")
    if sand.attrs['units'] == 'percent':
        sand = sand / 100.0
    if clay.attrs['units'] == 'percent':
        clay = clay / 100.0
    if silt.attrs['units'] == 'percent':
        silt = silt / 100.0

    # Check if sand, clay and silt add up to more than 1
    sum = sand + clay + silt
    if np.any(sum!=1.0):
        sand = sand / sum
        clay = clay / sum
        silt = silt / sum
        print('Clay + Sand + silt is not equal to 1')

    # Classification logic
    #1: SAND, 2: LOAMY SAND, 3: SANDY LOAM,
    #4: SILT LOAM, 5: SILT, 6: LOAM, 7: SANDY CLAY LOAM, 8: SILTY CLAY LOAM,
    #9: CLAY LOAM, 10: SANDY CLAY, 11: SILTY CLAY, 12:CLAY
    coords = {}
    for coord_name in sand.coords:
        coords[coord_name] = (list(sand.coords[coord_name].dims), sand[coord_name].values)
    dr1 = xr.DataArray(
        data=np.ones(sand.shape, dtype=np.int32)*-999,
        dims=list(sand.dims),
        coords=coords,
        attrs={'long_name':'usda soil class'}
    )
    # Classification logic
    dr1 = xr.where(silt+ 1.5 * clay < 0.15, 1, dr1)  # 1 'SAND'
    dr1 = xr.where((silt+ 1.5 * clay >= 0.15) & (silt+ 2 * clay < 0.3), 2, dr1) # 2 'LOAMY SAND'
    dr1 = xr.where(((clay >= 0.07) & (clay < 0.2) & (sand >0.52) & (silt+ 2.0 * clay >= 0.3)) | ((clay < 0.07) & (silt < 0.5) & (silt + 2 * clay >= 0.3)), 3, dr1) # 3 'SANDY LOAM'
    dr1 = xr.where(((clay >= 0.12) & (clay < 0.27) & (silt >= 0.50)) | ((silt >= 0.50) & (silt < 0.80) & (clay <= 0.12)), 4, dr1) # 4 'SILT LOAM'
    dr1 = xr.where((silt >= 0.8) & (clay < 0.12), 5, dr1) # 5 'SILT'
    dr1 = xr.where((clay >= 0.07) & (clay < 0.27) &  (silt >= 0.28) &  (silt < 0.5) & (sand <= 0.52), 6, dr1) # 6 'LOAM'
    dr1 = xr.where((clay >= 0.2) & (clay < 0.35) & (silt < 0.28) & (sand > 0.45), 7, dr1) # 7 'SANDY CLAY LOAM'
    dr1 = xr.where((clay >= 0.27) & (clay < 0.40) & (sand <= 0.20), 8, dr1) # 8 'SILTY CLAY LOAM'
    dr1 = xr.where((clay >= 0.27) & (clay < 0.40) & (sand > 0.20) & (sand <= 0.45), 9, dr1) # 9 'CLAY LOAM'
    dr1 = xr.where((clay >= 0.35) & (sand > 0.45), 10, dr1) # 10 'SANDY CLAY'
    dr1 = xr.where((clay >= 0.4) & (silt >= 0.4), 11, dr1) # 11 'SILTY CLAY'
    dr1 = xr.where((clay >= 0.4) & (sand <= 0.45) & (silt < 0.4), 12, dr1) # 12 'CLAY'

    return dr1


def plot_USDA_soil_triangle():
    """
    Plotting the USDA soil texture triangle
    """

    plt.figure()
    plt.plot([0, 100], [0, 0], 'k')
    plt.gca().invert_xaxis()
    plt.plot([0, 50], [0, 100], 'k')
    plt.plot([50, 100], [100, 0], 'k')
    plt.xlabel('Sand [%]')
    plt.text(95, 60, 'Clay [%]', fontsize=14)
    plt.text(25, 60, 'Silt [%]', fontsize=14)
    plt.text(55, -5, 'Sand [%]', fontsize=14)

    # Plot regions for the different soil classes
    plt.fill([100, 85, 95, 100], [0, 0, 10, 0], 'gray')
    plt.text(97, 3, 'Sand', color='white')

    plt.fill([85, 70, 92.4, 95, 85], [0, 0, 15, 10, 0], 'yellow')
    plt.text(87, 3, 'Loamy sand')

    plt.fill([70, 50, 46.7, 55, 62, 90, 92.4, 70], [0, 0, 7, 7, 20, 20, 15, 0], 'brown')
    plt.text(75, 10, 'Sandy Loam')

    plt.fill([55, 46.7, 38, 58.1, 62, 55], [7, 7, 27, 27, 20, 7], 'green')
    plt.text(55, 15, 'Loam')

    plt.fill([50, 20, 13.5, 6, 13.2, 38, 50], [0, 0, 12, 12, 27, 27, 0], 'olive')
    plt.text(35, 10, 'Silt Loam')

    plt.fill([20, 0, 6, 13.5, 20], [0, 0, 12, 12, 0], 'lightgreen')
    plt.text(12, 5, 'Silt')

    plt.fill([90, 62, 58.1, 62.5, 82.5, 90], [20, 20, 27, 35, 35, 20], 'red')
    plt.text(85, 27, 'Sandy Clay Loam')

    plt.fill([58.1, 33, 40, 65, 58], [27, 27, 40, 40, 27], 'blue')
    plt.text(55, 34, 'Clay Loam', color='white')

    plt.fill([33, 13.5, 20, 40, 33], [27, 27, 40, 40, 27], 'lightgray')
    plt.text(33, 33, 'Silty Clay Loam')

    plt.fill([82.5, 62.5, 72.5, 82.5], [35, 35, 55, 35], 'black')
    plt.text(80, 38, 'Sandy Clay', color='white')

    plt.fill([65, 40, 30, 50, 72.5, 65], [40, 40, 60, 100, 55, 40], 'orange')
    plt.text(55, 60, 'Clay')

    plt.fill([40, 20, 30, 40], [40, 40, 60, 40], 'darkgray')
    plt.text(35, 45, 'Silty Clay', color='white')

    plt.axis('off')

    # Plot data points
#    for i in range(len(sand)):
#        o = 100 - (1 - clay[i]) * 100  # offset
#        plt.plot(sand[i] * 100 + o / 2, clay[i] * 100, 'ok', markerfacecolor='white')
