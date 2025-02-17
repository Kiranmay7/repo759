#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 1
#SBATCH -J FirstSlurm_matmul
#SBATCH -o matmul.out 
#SBATCH -e matmul-%j.err
hostname
./task3 1024
