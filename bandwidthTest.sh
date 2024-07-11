#!/bin/bash
#SBATCH -N 16
#SBATCH -C gpu
#SBATCH -G 16
#SBATCH -q preempt
#SBATCH -J BandwidthTest
#SBATCH --mail-user=bc780@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:20:00
#SBATCH -A m4410_g

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500


#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py
srun -n 16 -c 128 --cpu_bind=cores -G 16 --gpu-bind=single:1  bandwidthTest.py