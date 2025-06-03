#!/usr/bin/env python
'''
remove data dimension element for no overlapping case
'''
import time
import sys
import os
import argparse
import xarray as xr
import numpy as np

# overlapping polygons (or grid box) variable names
#varname_overlap = 'overlapPolyId'
varname_overlap = 'intersector'

encodings={varname_overlap: {'dtype':'int64', '_FillValue':None},
           'IDmask':        {'dtype':'int32', '_FillValue':None}}

def process_command_line():
    '''Parse the commandline'''
    parser = argparse.ArgumentParser(description='Script to output intersecting element id and its weight for each target element')
    parser.add_argument('in_map',
                        help='input mapping netcdf')
    args = parser.parse_args()

    return(args)

# main
if __name__ == '__main__':

  args = process_command_line()

  ds = xr.open_dataset(args.in_map).load()

  # Check data dimension size and sum of overlapping elements (expected to be the same)
  # grid2poly.py output skip data in data dimension, but poly2poly put missing values in data dimension
  dim_data_size = ds.dims['data']
  sum_overlaps  = ds.overlaps.values.sum()
  flag = False
  if dim_data_size != sum_overlaps:
      flag = True
      print('WARNING: data dimension size (%d) is not equal to sum of overlapping elements (%d)'%(dim_data_size, sum_overlaps))

  ds_data = ds.drop_dims('polyid')
  ds_poly = ds.drop_dims('data')

  ds_data1 = ds_data.where(ds[varname_overlap]!=0,drop=True)

  # merge
  ds_mod = xr.merge([ds_poly,ds_data1])

  for varname, encoding in encodings.items():
      ds_mod[varname].encoding = encoding

  ds_mod.to_netcdf(args.in_map.replace('.nc','_mod.nc'))
