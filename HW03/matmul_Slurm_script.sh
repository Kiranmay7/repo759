#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 20
#SBATCH -J Slurm_matmul
#SBATCH -o matmul.out
#SBATCH -e matmul-%j.err
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp
hostname
for ((i = 1; i<=20; ++i))
do
  t=$i
  ./task1 1024 $t
done