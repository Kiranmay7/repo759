#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH -J reduce
#SBATCH -o reduce.out 
#SBATCH -e reduce.err
hostname
for ((i=1024; i<=1073741824; i*=2))
do
  n=$i
  ./task2 $n 1024
done
