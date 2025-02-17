#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 4
#SBATCH -J FirstSlurm_convolve
#SBATCH -o convolve.out 
#SBATCH -e convolve-%j.err
hostname
./task2 4 3
