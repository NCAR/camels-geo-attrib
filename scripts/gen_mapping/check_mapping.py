#!/usr/bin/env python
'''
output polygon id and overalpping grid ID, i-index, j-index, weight

grid ID is integer incrementing from LL longitudinally first
i-index: from west to east
i-index: from south to north

Recommend visual checking correct grid boxes are overlapping the polygon

'''
import time
import sys
import os
import argparse
import xarray as xr
import numpy as np

# mappring file: true: grid -> poly, false: poly->poly
grid2poly = False
# overlapping polygons (or grid box) variable names
#varname_overlap = 'overlapPolyId'
varname_overlap = 'intersector'

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

  ofile = os.path.splitext(args.in_map)[0] + ".asc"

  ds = xr.open_dataset(args.in_map).load()

  fout = open(ofile, "w")
  if (grid2poly):
    if varname_overlap in ds.variables:
      fout.write('seq polyid overlap_grid_id i_index j_index weight\n')
      #fout.write('seq polyid overlap_grid_id i_index j_index weight IDmask regridweight\n')
    else:
      fout.write('seq polyid i_index j_index weight\n')
  else:
    fout.write('seq polyid overlap_poly_id weight\n')

  # Check data dimension size and sum of overlapping elements (expected to be the same)
  # grid2poly.py output skip data in data dimension, but poly2poly put missing values in data dimension
  dim_data_size = ds.dims['data']
  sum_overlaps  = ds.overlaps.values.sum()
  flag = False
  if dim_data_size != sum_overlaps:
      flag = True
      print('WARNING: data dimension size (%d) is not equal to sum of overlapping elements (%d)'%(dim_data_size, sum_overlaps))

  ix2=0;
  for idx, poly in enumerate(ds.polyid):

     start = time.time()

     noverlap = ds.overlaps[idx].values

     if noverlap>0:
         ix1 = ix2
         ix2 = ix1+noverlap
     elif noverlap==0 and flag: # WARNNG: grid2poly.py output skip data in data dimension, but poly2poly put missing values in data dimension
         ix1 = ix2
         ix2 = ix2+1
     elif noverlap==0 and not flag:
         continue

     weight = ds.weight.values[ix1:ix2]
     regridweight = ds.regridweight.values[ix1:ix2]
     IDmask = ds.IDmask.values[ix1:ix2]
     if (grid2poly):
         if varname_overlap in ds.variables:
             overlapPolyId = ds[varname_overlap].values[ix1:ix2]
         i_index = ds.i_index.values[ix1:ix2]
         j_index = ds.j_index.values[ix1:ix2]
     else:
         overlapPolyId = ds[varname_overlap].values[ix1:ix2]

     # check sum of weight. should be 1
     sum_weight = weight.sum()
     if sum_weight-1.0 > 1e-5:
         print('%d: %.10f not equal to 1'%(poly, sum_weight))

     # output in text
     for jdx in np.arange(ix2-ix1):
         if (grid2poly):
             if varname_overlap in ds.variables:
                 fout.write('%d %d %d %d %d %.10f'%(idx, poly, overlapPolyId[jdx], i_index[jdx], j_index[jdx], weight[jdx]))
                 #fout.write('%d %d %d %d %d %.10f %d %.10f'%(idx, poly, overlapPolyId[jdx], i_index[jdx], j_index[jdx], weight[jdx], IDmask[jdx], regridweight[jdx]))
             else:
                 fout.write('%d %d %d %d %.10f'%(idx, poly, i_index[jdx], j_index[jdx], weight[jdx]))
         else:
             fout.write('%d %d %d %.10f'%(idx, poly, overlapPolyId[jdx], weight[jdx]))
         fout.write('\n')

     if idx%10000 == 0:
       elapsed = time.time()-start
       print('idx:%6d elapsed: %s sec'%(idx, elapsed))

  fout.close()
