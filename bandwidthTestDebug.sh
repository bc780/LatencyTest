#!/bin/bash
#SBATCH -N 4
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q debug
#SBATCH -J BandwidthTestDebug
#SBATCH --mail-user=bc780@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:01:00
#SBATCH -A m4410_g

#OpenMP settings:
module load python/3.11
module load pytorch/2.0.1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 4 -c 128 --cpu_bind=cores -G 4 --gpu-bind=none python bandwidthTest.py
