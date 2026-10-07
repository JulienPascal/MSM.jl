#!/bin/bash -
#SBATCH -J MSM
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH --qos=qos-batch
#SBATCH -o slurmMSM.out
#SBATCH -e slurmMSM.err

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

# Estimate on cluster (OnCluster is detected automatically inside a SLURM job;
# one worker per SLURM task)
julia $PWD/LinearModelCluster.jl
