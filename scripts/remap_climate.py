#!/usr/bin/env python
'''
    Remap forcing grid into arbitrary polygons using areal weight mean
'''
# =========================================================================
import sys, os, time
import glob
import argparse
import numpy as np
import xarray as xr
import pandas as pd

import utility as util

variables = ['sw_avg'  ,
             'lw_avg'  ,
             'prcp_avg',
             'tair_avg',
             'q_avg'   ,
             'uwnd_avg',
             'vwnd_avg',
             'pres_avg',
            ]

def process_command_line():
    '''Parse the commandline'''
    parser = argparse.ArgumentParser(description='remap climate time series data to HRU and write in netcdf')
    parser.add_argument('clim_dir', help='directory where climate netcdfs are located')
    parser.add_argument('wgt_nc',   help='path to mapping netcdf.')
    parser.add_argument('out_dir',  help='path to remapped forcing netcdf.')
    return parser.parse_args()


def getMeta(clim_nc):
    """ get variable attributes and encodings in netcdf """
    attrs = {}; encodings={}
    with xr.open_dataset(clim_nc) as ds:
        for varname in ds.variables:
           attrs[varname] = ds[varname].attrs
           encodings[varname] = ds[varname].encoding
    return attrs, encodings

if __name__ == '__main__':

    # process command line
    args = process_command_line()

    nclist = glob.glob(os.path.join(args.clim_dir, '*.nc'))

    # get input forcing variables attributes and encodings
#    var_attrs, var_encodings = getMeta( nclist[0] )

    # start reading input forcing data (dims of data variables: [Time, south_north, west_east]); keep file open
    for nc in nclist:
      xd = xr.open_dataset(nc)
      dr_mask = xr.where(np.isnan(xd['tair_avg'].isel(time=0)),0,1)
      a = util.regrid_mean_timeSeries(xr.open_dataset(args.wgt_nc), xd, dr_mask, variables )

      # write the output file
      out_nc = os.path.join(args.out_dir, os.path.basename(nc))
      a.to_netcdf(out_nc)
      print("wrote output file %s " % out_nc)
