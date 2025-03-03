#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 20
#SBATCH -J Slurm_msort
#SBATCH -o msort.out
#SBATCH -e msort-%j.err
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
hostname
for ((i =2**1; i<=2**10; i*=2))
do
  ts=$i
  ./task3 1000000 8 $ts
done