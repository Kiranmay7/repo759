#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 8
#SBATCH -J Slurm_nbody_task2
#SBATCH -o n_body_task2.out
#SBATCH -e n_body_task2-%j.err
g++ task2.cpp -Wall -O3 -std=c++17 -o task2
hostname
./task2 100 100