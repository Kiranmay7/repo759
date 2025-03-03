#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 20
#SBATCH -J Slurm_msort2
#SBATCH -o msort2.out
#SBATCH -e msort2-%j.err
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
hostname
for ((i=1; i<=20; ++i))
do
  t=$i
  ./task3 1000000 $t 256
done