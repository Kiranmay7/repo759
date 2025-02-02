#!/usr/bin/env zsh
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm-%j.out -e FirstSlurm-%j.err
#SBATCH -c 2
hostname