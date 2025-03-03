#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 20
#SBATCH -J Slurm_matmulconvolve
#SBATCH -o convolve.out
#SBATCH -e convolve-%j.err
hostname
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
for ((i = 1; i<=20; ++i))
do
  t=$i
  ./task2 1024 $t
done