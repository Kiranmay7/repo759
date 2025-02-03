#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 2
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out 
#SBATCH -e FirstSlurm-%j.err
hostname
./task6 6