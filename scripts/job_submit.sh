#!/bin/bash
#PBS -q casper 
#PBS -A NRAL0044 
#PBS -N remap 
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=100GB
#PBS -j oe 

#examples: 
#  qsub -v gcm=ACCESS-CM2,scen=historical,ens=r1i1p1f1 02_postprocess_ms.sh  

export TMPDIR=$SCRATCH/temp
mkdir -p $TMPDIR 

module load conda
conda activate npl-2024b

cd $PBS_O_WORKDIR 

./remap_climate.py /glade/u/home/mizukami/proj/cmip6_hydro/camels/attributes/ingredient/climate/nldas/sub4 ./gen_mapping/weight_file/spatialweights_nldas12km_to_CONUS_HUC12.nc /glade/derecho/scratch/mizukami/cmip6_hydro/camels/climate/nldas/CONUS_HUC12/
