#!/bin/bash
#SBATCH -N 4
#SBATCH -C gpu
#SBATCH -G 16
#SBATCH -q debug
#SBATCH -J reverseDebug
#SBATCH --mail-user=bc780@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00
#SBATCH -A m4431_g

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
echo $SLURM_NODELIST
srun -n 16 -c 32 --cpu_bind=cores -G 16 --gpu-bind=single:1 python reverseCheck.py
echo "BREAK"



