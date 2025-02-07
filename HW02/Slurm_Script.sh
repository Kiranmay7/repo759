#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -c 1
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out 
#SBATCH -e FirstSlurm-%j.err
hostname
for ((i=2**10; i<=2**30; i*=2))
do
  n=$i
  #echo $n
  ./task1 $n
done
