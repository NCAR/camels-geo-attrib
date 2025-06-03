#!/bin/bash -l
#PBS -A P48500028
#PBS -q casper 
#PBS -N p2p 
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -l select=1:ncpus=36:mpiprocs=36

export TMPDIR=$SCRATCH/temp
mkdir -p $TMPDIR 

cd $PBS_O_WORKDIR

module load conda
conda activate npl-2024b

Polygon1=./geospatial/gagesII_671_shp_geogr.gpkg
FieldName1=GAGE_ID  # HUCIDXint for conus_HUC12_merit_v7b.gpkg,  hru_id for HCDN_nhru_final_671.buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level1.gpkg
#Polygon2=./geospatial/STATSGO_GPKG_WGS84_KMS.gpkg
#FieldName2=poly_id_us
#outnc=./weight_file/spatialweights_STATSGO_to_camels.nc
#Polygon2=./geospatial/GLHYMPS_North_America_WGS84.gpkg
#FieldName2=OBJECTID
#outnc=./weight_file/spatialweights_GLHYMPS_to_camels.nc
Polygon2=./geospatial/LiMW2015_North_America_WGS84.gpkg
FieldName2=OBJECTID
outnc=./weight_file/spatialweights_LiMW2015_to_camels.nc

python ./poly2poly.py $Polygon1 $FieldName1 $Polygon2 $FieldName2 $outnc

'''
      <Polygon1>     -> target polygon: Polygon GeoPackage (full path)
      <FieldName1>   -> Polygon identifier field name for Polygon1
      <Polygon2>     -> source polygon: Polygon GeoPackage2 (full path)
      <FieldName2>   -> Polygon identifier field name for Polygon 2
      <gridflag>     -> Optional - Type 'GRID' to indicate Polygon2 was generated from grid2shp.py
      <outputNC>     -> Optional - Full path to output netCDF file (.nc extentsion required)
'''
