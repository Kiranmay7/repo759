#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 8
#SBATCH -J Slurm_task3
#SBATCH -o task3.out
#SBATCH -e task3-%j.err
g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
hostname
./task3 100 100 4