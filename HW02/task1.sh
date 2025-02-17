#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 1
#SBATCH -J FirstSlurm
#SBATCH -o task1.out 
#SBATCH -e task1-%j.err
hostname
for ((i=2**10; i<=2**30; i*=2))
do
  n=$i
  #echo $n
  ./task1 $n
done
