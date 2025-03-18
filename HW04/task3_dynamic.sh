#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 8
#SBATCH -J Slurm_task3_dynamic
#SBATCH -o task3_dynamic.out
#SBATCH -e task3_dynamic-%j.err
g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
hostname
for ((i=1;i<=8;i++))
do
  ts=$i
  ./task3 100 100 $ts
done