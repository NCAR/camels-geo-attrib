#!/bin/bash -l
#PBS -A P48500028
#PBS -q casper 
#PBS -N g2p 
#PBS -l walltime=0:50:00
#PBS -j oe
#PBS -l select=1:ncpus=24:mpiprocs=24

export TMPDIR=$SCRATCH/temp
mkdir -p $TMPDIR 

cd $PBS_O_WORKDIR

module load conda
conda activate npl-2024b

python ./grid2poly.py 
