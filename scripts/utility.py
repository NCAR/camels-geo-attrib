#!/usr/bin/env python

import os
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import fiona
from collections import defaultdict

# in case toml module may not be available
try:
    import tomli
    def load_toml(toml_file) -> dict:
        """Load TOML data from file """
        with open(toml_file, 'rb') as f:
            return tomli.load(f)
except ImportError:
    pass #or anything to log


def load_yaml(yaml_file) -> dict:
    """Load yaml data from file """
    with open(yaml_file, "r") as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.FullLoader)


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

# ---- geopackage reading
def records(filename, usecols, **kwargs):
    with fiona.open(filename, 'r') as src:
        for feature in src:
            f = {k: feature[k] for k in ['id', 'geometry']}
            f['properties'] = {k: feature['properties'][k] for k in usecols}
            yield f

def read_shps(gpkg_list, usecols, **kwargs):
    gdf_frame = []
    for gpkg in gpkg_list:
        gdf_frame.append(gpd.GeoDataFrame.from_features(records(gpkg.strip('\n'), usecols, **kwargs)))
        print('Finished reading %s'%gpkg.strip('\n'))
    return pd.concat(gdf_frame)

try:
    from pyogrio import read_dataframe
    def read_gpkg(gpkg_list,  usecols, **kwargs):
        """Load geopackage with selected attributes in dataframe"""
        gdf_frame = []
        for gpkg in gpkg_list:
            gdf_frame.append(read_dataframe(gpkg, columns=usecols))
            print('Finished reading %s'%gpkg.strip('\n'))
        return pd.concat(gdf_frame)
except ImportError:
    pass #or anything to log

def mode_func(array):
    values, counts = np.unique(array, return_counts=True)
    return values[np.argmax(counts)]

# ---- general functions
def find_dominant(values: np.array, weight: np.array):
    '''
    Calculate total weight for each unique value in values array then find the 1st and 2nd dominant value and their weight

      e.g.,
      values = np.array([1,2,2,3,6,6])
      weight = np.array([0.1,0.03,0.02,0.4,0.25,0.2])

      sorted_values = find_dominant(values, weight)
      sorted_values = [(6, 0.45), (3, 0.4), (1, 0.1), (2, 0.05)] # each tuple in sorted list is (value, weight)
    '''
    #
    value_weight_map = defaultdict(float)
    for val, w in zip(values, weight):
        value_weight_map[val] += w

    # Sort values by weight in descending order
    sorted_values = sorted(value_weight_map.items(), key=lambda x: x[1], reverse=True)

    # Most dominant and second dominant elements
    #most_dominant = sorted_values[0]
    #second_dominant = sorted_values[1]

    return sorted_values


def get_index_array(a_array, b_array):
    '''
    Get index array where each index points to locataion in a_array. The order of index array corresponds to b_array

      e.g.,
      a_array = [2, 4, 1, 8, 3, 10, 5, 9, 7, 6]
      b_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      result  = [2, 0, 4, 1, 6, 9, 8, 3, 7, 5]

    https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    '''
    index = np.argsort(a_array)
    sorted_a_array = a_array[index]
    sorted_index = np.searchsorted(sorted_a_array, b_array)

    yindex = np.take(index, sorted_index, mode="clip")
    mask = a_array[yindex] != b_array

    result = np.ma.array(yindex, mask=mask)

    return result


def reorder_index(ID_array_orig, ID_array_target):
    x = ID_array_orig
    # Find the indices of the reordered array
    # From: https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, ID_array_target)

    return np.take(index, sorted_index, mode="clip")

# Function to map xarray data array containing dictionary key to dictionary values
def map_param(da, dictionary, key):
    # Function to handle NaN or map tree types
    def safe_map(x):
        if np.isnan(x):  # If x is NaN, return NaN
            return np.nan
        return dictionary[x][key]  # Otherwise, map from root_param

    param_array = np.vectorize(safe_map)(da.values)
    return xr.DataArray(param_array, dims=da.dims, coords=da.coords, name=key)

#---- netcdf related functions
def no_time_variable(ds: xr.DataArray):
    """ get variable without time dimension"""
    vars_without_time = []
    for var in ds.variables:
        if 'time' not in ds[var].dims:
            if var not in list(ds.coords):
                vars_without_time.append(var)
    return vars_without_time

def get_netCDF_data(fn):
    """ get dataset"""
    return xr.open_dataset(fn)#.chunk({'lat':150,'lon':360})

def get_netCDF_value(ds, varname):
    """Read <varname> variables from ds """
    var_data = ds[varname].values
    return var_data

def get_netCDF_var_ndim(ds, varname):
    """Read ndim (array rank) of <varname> variables in xarray dataset ds """
    return ds[varname].ndim

def get_netCDF_attr(nc):
    """ get variable attributes and encodings in netcdf """
    attrs = {}; encodings={}
    with xr.open_dataset(nc) as ds:
        for varname in ds.variables:
           attrs[varname] = ds[varname].attrs
           encodings[varname] = ds[varname].encoding
    return attrs, encodings


# ---- Aggregation
def remap_mean(ds_weight: xr.Dataset, xd: xr.Dataset, dr_mask: xr.DataArray, varname_list) -> xr.Dataset:
    """
       Compute areal weighted avg value of <varname> in <nc_in> for all output polygons
       based on input/output overlap weights in <nc_wgt>
    """

    # Get data from spatial weights file
    wgtVals = ds_weight['weight'].values
    intersectors = ds_weight['intersector'].values
    allOutPolyIDs = ds_weight['polyid'].values
    overlaps = ds_weight['overlaps'].values

    ix_hru = {h: i for i, h in enumerate(xd['hru'].values)}

    ix_intersector = np.ones(len(intersectors), dtype='int32') *-999
    for ix,value in enumerate(intersectors):
      ix_intersector[ix] = ix_hru[value]

    # mask weight at source grid point without valid data
    # mask assums 0 or nan indicates outside land
    for ii, jx in enumerate(ix_intersector):
        if dr_mask.values[jx]==0 or np.isnan(dr_mask.values[jx]):
            wgtVals[ii] = 0.0

    # check run end limits and calculate some useful dimensions
    nOutPolys = len(allOutPolyIDs)
    print("Averaging areas for %d polygons: " % nOutPolys)
    maxOverlaps = overlaps.max()

    # calculate indices for subsetting polygon range (for all data)
    overlapEndNdx   = np.cumsum(overlaps)
    overlapStartNdx = np.cumsum(overlaps)-overlaps

    # assign weights and indices to a regular array (nOutPolys x maxOverlaps)
    matWgts = np.zeros((nOutPolys, maxOverlaps))
    for p in range(0, nOutPolys):
        wgt_array = wgtVals[overlapStartNdx[p]:overlapEndNdx[p]]
        weight_sum = np.sum(wgt_array)
        if weight_sum>0:
            matWgts[p, 0:overlaps[p]] = wgt_array/weight_sum
        else:
            matWgts[p, :] = np.nan

    # now use these weights in the avg val computation
    dataset = xr.Dataset()
    for var_name in varname_list:
        # now read the input timeseries file
        dataVals = xd[var_name].fillna(0.0).values

        matDataVals = np.zeros((nOutPolys, maxOverlaps))
        wgtedVals   = np.zeros((nOutPolys))   # commented out on 12/13

        # reformat var data into regular matrix matching weights format (nOutPolygons, maxOverlaps)
        #   used advanced indexing to extract matching input grid indices
        for p in range(0, nOutPolys):
                matDataVals[p, 0:overlaps[p]] = \
                    dataVals[[ix_intersector[overlapStartNdx[p]:overlapEndNdx[p]]] ]

        wgtedVals = np.sum(matDataVals * matWgts, axis=1)   # produces vector of weighted values

        dataset[var_name] = xr.DataArray(data=wgtedVals,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        print("-------------------")
        print("  averaged %s" % (var_name))

    return dataset


def remap_mean_timeSeries(ds_weight: xr.Dataset, xd: xr.Dataset, dr_mask: xr.DataArray, varname_list, time_name='time') -> xr.Dataset:
    """
       Compute areal weighted avg value of <varname> in <nc_in> for all output polygons
       based on input/output overlap weights in <nc_wgt>
    """

    # Get data from spatial weights file
    wgtVals = ds_weight['weight'].values
    intersectors = ds_weight['intersector'].values
    allOutPolyIDs = ds_weight['polyid'].values
    overlaps = ds_weight['overlaps'].values

    ix_hru = {h: i for i, h in enumerate(xd['hru'].values)}

    ix_intersector = np.ones(len(intersectors), dtype='int32') *-999
    for ix,value in enumerate(intersectors):
      ix_intersector[ix] = ix_hru[value]

    # read times variables
    times = {time_name: xd[time_name]}
    nTimeSteps = len(times[time_name])

    # mask weight at source grid point without valid data
    # mask assums 0 or nan indicates outside land
    for ii, jx in enumerate(ix_intersector):
        if dr_mask.values[jx]==0 or np.isnan(dr_mask.values[jx]):
            wgtVals[ii] = 0.0

    # check run end limits and calculate some useful dimensions
    nOutPolys = len(allOutPolyIDs)
    print("Averaging areas for %d polygons: " % nOutPolys)
    maxOverlaps = overlaps.max()

    # calculate indices for subsetting polygon range (for all data)
    overlapEndNdx   = np.cumsum(overlaps)
    overlapStartNdx = np.cumsum(overlaps)-overlaps

    # assign weights and indices to a regular array (nOutPolys x maxOverlaps)
    matWgts = np.zeros((nOutPolys, maxOverlaps))
    for p in range(0, nOutPolys):
        weight_sum = wgtVals[overlapStartNdx[p]:overlapEndNdx[p]].sum()
        if weight_sum>0:
            matWgts[p, 0:overlaps[p]] = wgtVals[overlapStartNdx[p]:overlapEndNdx[p]]/weight_sum
        else:
            matWgts[p, :] = np.nan

    # now use these weights in the avg val computation
    dataset = xr.Dataset()
    for var_name in varname_list:
        # now read the input timeseries file
        dataVals = xd[var_name].fillna(0.0).values

        matDataVals = np.zeros((nOutPolys, maxOverlaps))
        wgtedVals   = np.zeros((nTimeSteps, nOutPolys))   # commented out on 12/13

        for t in range(0, nTimeSteps):

            # reformat var data into regular matrix matching weights format (nOutPolygons, maxOverlaps)
            #   used advanced indexing to extract matching input grid indices
            for p in range(0, nOutPolys):
                matDataVals[p, 0:overlaps[p]] = \
                    dataVals[t, [ix_intersector[overlapStartNdx[p]:overlapEndNdx[p]]] ]

            wgtedVals[t, ] = np.sum(matDataVals * matWgts, axis=1)   # produces vector of weighted values

        dataset[var_name] = xr.DataArray(data=wgtedVals,
                     dims=[time_name, "hru"],
                     coords=dict(time=xd[time_name], hru=allOutPolyIDs)
                    )
        print("-------------------")
        print("  averaged %s for %d timesteps" % (var_name, nTimeSteps))

    return dataset


def regrid_mean_timeSeries(ds_weight: xr.Dataset, xd: xr.Dataset, dr_mask: xr.DataArray, varname_list, time_name='time') -> xr.Dataset:
    """
    Compute areal weighted avg value of <varname> in <nc_in> for all output polygons
       based on input/output overlap weights in <nc_wgt>
    """

    # Get data from spatial weights file
    wgtVals = ds_weight['weight'].values
    i_index = ds_weight['i_index'].values
    j_index = ds_weight['j_index'].values        # j_index --- LAT direction
    allOutPolyIDs = ds_weight['polyid'].values
    overlaps = ds_weight['overlaps'].values

    # read times variables
    times = {time_name: xd[time_name]}
    nTimeSteps = len(times[time_name])

    # set to zero based index and flip the N-S index (for read-in data array 0,0 is SW-corner)
    j_index = j_index - 1              # j_index starts at North, and at 1 ... make 0-based
    i_index = i_index - 1              # i_index starts at West, and a 1 ... make 0-based

    # mask weight at source grid point without valid data
    # mask assums 0 or nan indicates outside land
    for ii, (ix, jx) in enumerate(zip(i_index, j_index)):
        if dr_mask.values[jx,ix]==0 or np.isnan(dr_mask.values[jx,ix]):
            wgtVals[ii] = 0.0

    # check run end limits and calculate some useful dimensions
    nOutPolys = len(allOutPolyIDs)
    print("Averaging %d vriables for %d polygons: " % (len(varname_list), nOutPolys))
    maxOverlaps = overlaps.max()

    # calculate indices for subsetting polygon range (for all data)
    overlapEndNdx   = np.cumsum(overlaps)
    overlapStartNdx = np.cumsum(overlaps)-overlaps

    # assign weights and indices to a regular array (nOutPolys x maxOverlaps)
    matWgts = np.zeros((nOutPolys, maxOverlaps))
    for p in range(0, nOutPolys):
        wgt_array = wgtVals[overlapStartNdx[p]:overlapEndNdx[p]]
        weight_sum = np.sum(wgt_array)
        if weight_sum>0:
            matWgts[p, 0:overlaps[p]] = wgt_array/weight_sum
        else:
            matWgts[p, :] = np.nan

    # now use these weights in the avg val computation
    dataset = xr.Dataset()
    for var_name in varname_list:
        # now read the input timeseries file
        dataVals = xd[var_name].fillna(0.0).values

        matDataVals = np.zeros((nOutPolys, maxOverlaps))
        wgtedVals   = np.zeros((nTimeSteps, nOutPolys))   # commented out on 12/13

        # format data into shape matching weights [nOutPolys, maxOverlaps]
        for t in range(0, nTimeSteps):

            # reformat var data into regular matrix matching weights format (nOutPolygons, maxOverlaps)
            #   used advanced indexing to extract matching input grid indices
            for p in range(0, nOutPolys):
                matDataVals[p, 0:overlaps[p]] = \
                    dataVals[t, [j_index[overlapStartNdx[p]:overlapEndNdx[p]]], \
                                [i_index[overlapStartNdx[p]:overlapEndNdx[p]]] ]

            wgtedVals[t, ] = np.sum(matDataVals * matWgts, axis=1)   # produces vector of weighted values

        dataset[var_name] = xr.DataArray(data=wgtedVals,
                     dims=[time_name, "hru"],
                     coords={time_name:xd[time_name], "hru":allOutPolyIDs}
                    )
        print("-------------------")
        print("  averaged %s for %d timesteps" % (var_name, nTimeSteps))

    return dataset

def regrid_mean(ds_weight: xr.Dataset, xd: xr.Dataset, dr_mask: xr.DataArray, varname_list, verbose=True) -> xr.Dataset:
    """
    Compute areal weighted avg value of <varname> in <nc_in> for all output polygons
       based on input/output overlap weights in <nc_wgt>
    """

    # Get data from spatial weights file
    wgtVals = ds_weight['weight'].values
    i_index = ds_weight['i_index'].values
    j_index = ds_weight['j_index'].values        # j_index --- LAT direction
    allOutPolyIDs = ds_weight['polyid'].values
    overlaps = ds_weight['overlaps'].values

    # set to zero based index and flip the N-S index (for read-in data array 0,0 is SW-corner)
    j_index = j_index - 1              # j_index starts at North, and at 1 ... make 0-based
    i_index = i_index - 1              # i_index starts at West, and a 1 ... make 0-based

    # mask weight at source grid point without valid data
    # mask assums 0 or nan indicates outside land
    for ii, (ix, jx) in enumerate(zip(i_index, j_index)):
        if dr_mask.values[jx,ix]==0 or np.isnan(dr_mask.values[jx,ix]):
            wgtVals[ii] = 0.0

    # check run end limits and calculate some useful dimensions
    nOutPolys = len(allOutPolyIDs)
    if verbose:
        print("Averaging %d vriables for %d polygons: " % (len(varname_list), nOutPolys))
    maxOverlaps = overlaps.max()

    # calculate indices for subsetting polygon range (for all data)
    overlapEndNdx   = np.cumsum(overlaps)
    overlapStartNdx = np.cumsum(overlaps)-overlaps

    # assign weights and indices to a regular array (nOutPolys x maxOverlaps)
    matWgts = np.zeros((nOutPolys, maxOverlaps))
    for p in range(0, nOutPolys):
        wgt_array = wgtVals[overlapStartNdx[p]:overlapEndNdx[p]]
        weight_sum = np.sum(wgt_array)
        if weight_sum>0:
            matWgts[p, 0:overlaps[p]] = wgt_array/weight_sum
        else:
            matWgts[p, :] = np.nan

    # now use these weights in the avg val computation
    dataset = xr.Dataset()
    for var_name in varname_list:
        # now read the input timeseries file
        dataVals = xd[var_name].fillna(0.0).values

        matDataVals = np.zeros((nOutPolys, maxOverlaps))
        wgtedVals   = np.zeros(nOutPolys)   # commented out on 12/13

        # format data into shape matching weights [nOutPolys, maxOverlaps]
        # reformat var data into regular matrix matching weights format (nOutPolygons, maxOverlaps)
        #   used advanced indexing to extract matching input grid indices
        for p in range(0, nOutPolys):
            matDataVals[p, 0:overlaps[p]] = \
                dataVals[[j_index[overlapStartNdx[p]:overlapEndNdx[p]]], \
                            [i_index[overlapStartNdx[p]:overlapEndNdx[p]]] ]

        wgtedVals = np.nansum(matDataVals * matWgts, axis=1)   # produces vector of weighted values
        #wgtedVals[t, ] = np.nansum(matDataVals * matWgts, axis=1)   # produces vector of weighted values

        dataset[var_name] = xr.DataArray(data=wgtedVals,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        if verbose:
            print("-------------------")
            print("  averaged %s" % (var_name))

    return dataset


def remap_mode(ds_weight: xr.Dataset, xd: xr.Dataset, dr_mask: xr.DataArray, varname_list, verbose=True) -> xr.Dataset:
    """
       Compute areal weighted avg value of <varname> in <nc_in> for all output polygons
       based on input/output overlap weights in <nc_wgt>
    """

    # Get data from spatial weights file
    wgtVals = ds_weight['weight'].values
    intersectors = ds_weight['intersector'].values
    allOutPolyIDs = ds_weight['polyid'].values
    overlaps = ds_weight['overlaps'].values
        
    ix_hru = {h: i for i, h in enumerate(xd['hru'].values)}

    ix_intersector = np.ones(len(intersectors), dtype='int32') *-999
    for ix,value in enumerate(intersectors):
      ix_intersector[ix] = ix_hru[value] 
        
    # mask weight at source grid point without valid data
    # mask assums 0 or nan indicates outside land
    for ii, jx in enumerate(ix_intersector):
        if dr_mask.values[jx]==0 or np.isnan(dr_mask.values[jx]):
            wgtVals[ii] = 0.0 
       
    # check run end limits and calculate some useful dimensions
    nOutPolys = len(allOutPolyIDs)
    if verbose:
        print("Averaging areas for %d polygons: " % nOutPolys)
    maxOverlaps = overlaps.max()

    # calculate indices for subsetting polygon range (for all data)
    overlapEndNdx   = np.cumsum(overlaps)
    overlapStartNdx = np.cumsum(overlaps)-overlaps

    # assign weights and indices to a regular array (nOutPolys x maxOverlaps)
    matWgts = np.zeros((nOutPolys, maxOverlaps))
    for p in range(0, nOutPolys):
        wgt_array = wgtVals[overlapStartNdx[p]:overlapEndNdx[p]]
        weight_sum = np.sum(wgt_array)
        if weight_sum>0:
            matWgts[p, 0:overlaps[p]] = wgt_array/weight_sum
        else:
            matWgts[p, :] = np.nan

    # now use these weights in the avg val computation
    dataset = xr.Dataset()
    for var_name in varname_list:
        # now read the input data
        dataVals = xd[var_name].values

        # initialize
        first_dominant = np.full((nOutPolys), 'N/A', np.object_)   # this is final aggregated data
        first_weight = np.ones((nOutPolys), 'float')*np.nan   # this is weight of final aggregated data
        second_dominant = np.full((nOutPolys), 'N/A', np.object_)   # this is final aggregated data
        second_weight = np.ones((nOutPolys), 'float')*np.nan   # this is weight of final aggregated data

        # reformat var data into regular matrix matching weights format (nOutPolygons, maxOverlaps)
        #   used advanced indexing to extract matching input grid indices
        for p in range(0, nOutPolys):
            sorted_data = find_dominant(dataVals[ix_intersector[overlapStartNdx[p]:overlapEndNdx[p]]],
                                        matWgts[p,0:overlaps[p]]) # see utilty.py for output data structure

            if len(sorted_data) > 0:
                first_dominant[p] = sorted_data[0][0] # 1st element in the 1st tuple
                first_weight[p] = sorted_data[0][1]   # 2nd element in the 1st tuple
                if len(sorted_data) > 1:
                    second_dominant[p] = sorted_data[1][0] # 1st element in the 2nd tuple
                    second_weight[p] = sorted_data[1][1]  # 2nd element in the 2nd tuple

        dataset[f'1st_dominant_{var_name}'] = xr.DataArray(data=first_dominant,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        dataset[f'2nd_dominant_{var_name}'] = xr.DataArray(data=second_dominant,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        dataset[f'1st_dominant_{var_name}_fraction'] = xr.DataArray(data=first_weight,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        dataset[f'2nd_dominant_{var_name}_fraction'] = xr.DataArray(data=second_weight,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        if verbose:
            print("-------------------")
            print("  averaged %s" % (var_name))

    return dataset

def regrid_mode(ds_weight: xr.Dataset, xd: xr.Dataset, dr_mask: xr.DataArray, varname_list, verbose=True) -> xr.Dataset:
    """
       Compute areal weighted avg value of <varname> in <nc_in> for all output polygons
       based on input/output overlap weights in <nc_wgt>
    """

    # Get data from spatial weights file
    wgtVals = ds_weight['weight'].values
    i_index = ds_weight['i_index'].values
    j_index = ds_weight['j_index'].values        # j_index --- LAT direction
    allOutPolyIDs = ds_weight['polyid'].values
    overlaps = ds_weight['overlaps'].values

    # set to zero based index and flip the N-S index (for read-in data array 0,0 is SW-corner)
    j_index = j_index - 1              # j_index starts at North, and at 1 ... make 0-based
    i_index = i_index - 1              # i_index starts at West, and a 1 ... make 0-based
            
    # mask weight at source grid point without valid data
    # mask assums 0 or nan indicates outside land
    for ii, (ix, jx) in enumerate(zip(i_index, j_index)):
        if dr_mask.values[jx,ix]==0 or np.isnan(dr_mask.values[jx,ix]):
            wgtVals[ii] = 0.0
            
    # check run end limits and calculate some useful dimensions
    nOutPolys = len(allOutPolyIDs)
    if verbose:
        print("Averaging areas for %d polygons: " % nOutPolys)
    maxOverlaps = overlaps.max()

    # calculate indices for subsetting polygon range (for all data)
    overlapEndNdx   = np.cumsum(overlaps)
    overlapStartNdx = np.cumsum(overlaps)-overlaps

    # assign weights and indices to a regular array (nOutPolys x maxOverlaps)
    matWgts = np.zeros((nOutPolys, maxOverlaps))
    for p in range(0, nOutPolys):
        wgt_array = wgtVals[overlapStartNdx[p]:overlapEndNdx[p]]
        weight_sum = np.sum(wgt_array)
        if weight_sum>0:
            matWgts[p, 0:overlaps[p]] = wgt_array/weight_sum
        else:
            matWgts[p, :] = np.nan

    # now use these weights in the avg val computation
    dataset = xr.Dataset()
    for var_name in varname_list:
        # now read the input data
        dataVals = xd[var_name].values

        # initialize
        first_dominant = np.full((nOutPolys), 'N/A', np.object_)   # this is final aggregated data
        first_weight = np.ones((nOutPolys), 'float')*np.nan   # this is weight of final aggregated data
        second_dominant = np.full((nOutPolys), 'N/A', np.object_)   # this is final aggregated data
        second_weight = np.ones((nOutPolys), 'float')*np.nan   # this is weight of final aggregated data

        # reformat var data into regular matrix matching weights format (nOutPolygons, maxOverlaps)
        #   used advanced indexing to extract matching input grid indices
        for p in range(0, nOutPolys):
            dataVals_sub = dataVals[ [j_index[overlapStartNdx[p]:overlapEndNdx[p]]], \
                            [i_index[overlapStartNdx[p]:overlapEndNdx[p]]] ][0]
            sorted_data = find_dominant(dataVals_sub, matWgts[p,0:overlaps[p]]) # see utilty.py for output data structure
            if len(sorted_data) > 0:
                first_dominant[p] = sorted_data[0][0] # 1st element in the 1st tuple
                first_weight[p] = sorted_data[0][1]   # 2nd element in the 1st tuple
                if len(sorted_data) > 1:
                    second_dominant[p] = sorted_data[1][0] # 1st element in the 2nd tuple
                    second_weight[p] = sorted_data[1][1]  # 2nd element in the 2nd tuple

        dataset[f'1st_dominant_{var_name}'] = xr.DataArray(data=first_dominant,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        dataset[f'2nd_dominant_{var_name}'] = xr.DataArray(data=second_dominant,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        dataset[f'1st_dominant_{var_name}_fraction'] = xr.DataArray(data=first_weight,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        dataset[f'2nd_dominant_{var_name}_fraction'] = xr.DataArray(data=second_weight,
                     dims=["hru"],
                     coords=dict(hru=allOutPolyIDs)
                    )
        if verbose:
            print("-------------------")
            print("  averaged %s" % (var_name))

    return dataset


def remap_mean_vertical(mapping_data, dr:xr.Dataset, pvalue=1,  default=-9999) -> xr.DataArray:
    """
    Compute areal weighted generalized mean value of for vertical layer.

    Note:
    numpy broadcasting rule
    https://numpy.org/doc/stable/user/basics.broadcasting.html

    ogirinal grid: 3D [lat, lon, soil_lyr] -> target grid: 3D [lat, lon, model_lyr]
                   2D [hru, soil_lyr] -> target grid: 3D [hru, model_lyr]
    """

    origArrays = dr.values
    orgArrays_reshaped = np.moveaxis(origArrays, 0, -1)
    array_shape        = orgArrays_reshaped.shape  # original array [soil_lyr, lat, lon] -> [lat, lon, soil_lyr] or [soil_lyr, hru] -> [hru, soil_lyr]
    nDims              = len(array_shape)

    # TODO
    # move these out of function
    nMlyr       = len(mapping_data['overlaps'])
    maxOverlaps = mapping_data['overlaps'].max()

    if nDims == 3:
        wgtedVals   = np.zeros((array_shape[0], array_shape[1], nMlyr), dtype='float32')
        matDataVals = np.full((array_shape[0], array_shape[1], nMlyr, maxOverlaps), np.nan, dtype='float32')
    elif nDims == 2:
        wgtedVals   = np.zeros((array_shape[0],  nMlyr), dtype='float32')
        matDataVals = np.full((array_shape[0],  nMlyr, maxOverlaps), np.nan, dtype='float32')
    else:
        pass # add error check - array with other dimension is not supported.

    # reformat var data into regular matrix matching weights format (nOutPolygons, maxOverlaps)
    #   used advanced indexing to extract matching input grid indices
    for ixm in range(0, nMlyr):
        if mapping_data['overlaps'][ixm]>0:
            if nDims == 3:
                matDataVals[:, :, ixm, 0:mapping_data['overlaps'][ixm]] = \
                    orgArrays_reshaped[:, :, mapping_data['soil_index'][ixm,0:mapping_data['overlaps'][ixm]]]
            elif nDims == 2:
                matDataVals[:, ixm, 0:mapping_data['overlaps'][ixm]] = \
                    orgArrays_reshaped[:, mapping_data['soil_index'][ixm,0:mapping_data['overlaps'][ixm]]]
        else:
            if nDims == 3:
                matDataVals[:, :, ixm, 0] = default
            elif nDims == 3:
                matDataVals[:, ixm, 0] = default
    if nDims == 3:
        weight = np.broadcast_to(mapping_data['weight'], (array_shape[0], array_shape[1], *mapping_data['weight'].shape ))
    elif nDims == 2:
        weight = np.broadcast_to(mapping_data['weight'], (array_shape[0], *mapping_data['weight'].shape ))
    if abs(pvalue) < 0.00001: # geometric mean
        wgtedVals = exp(_nansum(log(matDataVals)* weight, axis=nDims))
    else:
        wgtedVals = _nansum(matDataVals**pvalue * weight, axis=nDims) **(1.0/pvalue)   # produces vector of weighted values

    coords = {}
    for coord_name in dr.coords:
        coords[coord_name] = (list(dr.coords[coord_name].dims), dr[coord_name].values)
    attrs = dr.attrs
    dataarray = xr.DataArray(
        data=np.moveaxis(wgtedVals, -1, 0),
        dims=list(dr.dims),
        coords=coords,
        attrs=attrs
    )

    return dataarray


def _nansum(a, **kwargs):
    mx = np.isnan(a).all(**kwargs)
    res = np.nansum(a, **kwargs)
    res[mx] = np.nan
    return res


def comp_layer_weight(raw_thickness, target_thickness):

    #-- Compute for depths to 1)top and 2) bottom of soil and model layer

    nMlyr = len(target_thickness)

    soil_bot = np.cumsum(raw_thickness)
    model_bot = np.cumsum(target_thickness)

    model_top = shift(model_bot, 1, fill_value=0)
    soil_top  = shift(soil_bot, 1, fill_value=0)

    #-- Compute index of soil layer where top of model layer is located
    idxTop = np.zeros(len(target_thickness), dtype='int32')
    idxBot = np.zeros(len(target_thickness), dtype='int32')

    for ixm in range(nMlyr):
        for ixs in range(len(raw_thickness)):
            if model_top[ixm] < soil_bot[ixs]:
                idxTop[ixm] = ixs
                break

    #-- Compute index of soil layer where the bottom of model layer is located
    for ixm in range(nMlyr):
        for ixs in range(len(raw_thickness)):
            if model_bot[ixm] <= soil_bot[ixs]:
                idxBot[ixm] = ixs
                break

    maxSlyr = (idxBot-idxTop+1).max()
    mapping_data = {'overlaps':   np.zeros(nMlyr, dtype='int32'),
                    'soil_index': np.zeros((nMlyr, maxSlyr), dtype='int32'),
                    'weight' :    np.zeros((nMlyr, maxSlyr), dtype='float32'),
                   }

    for ixm in range(len(target_thickness)):

        if idxTop[ixm] == idxBot[ixm]: # if model layer is completely within soil layer
            mapping_data['overlaps'][ixm]      = 1
            mapping_data['soil_index'][ixm, 0] = ixs
            mapping_data['weight'][ixm, 0]     = 1.0
            continue

        # loop frm the upper most soil layer to the lowest soil layer that intersect current model layer
        counter = 0
        for ixs in range(idxTop[ixm], idxBot[ixm]+1):
            # if model layer contains multiple soil layers
            if ixs == idxTop[ixm]:           # for the upper most soil layer that intersect model layer
                mapping_data['weight'][ixm, counter] = (soil_bot[ixs] - model_top[ixm])/target_thickness[ixm]
            elif ixs == idxBot[ixm]:         # for the lowest soil layer that intersect model layer
                mapping_data['weight'][ixm, counter] = (model_bot[ixm] - soil_top[ixs])/target_thickness[ixm]
            else:                            # for soil layers that completely in model layer
                mapping_data['weight'][ixm, counter] = raw_thickness[ixs]/target_thickness[ixm]
            mapping_data['soil_index'][ixm, counter] = ixs
            counter += 1
        mapping_data['overlaps'][ixm]      = counter

    return mapping_data


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result
