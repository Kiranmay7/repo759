#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --gres=gpu:4
#SBATCH -J matmul
#SBATCH -o matmul.out
#SBATCH -e matmul-%j.err
hostname
for ((i=2**5; i<=2**14; i*=2))
do
  n=$i
  ./task1 $n 1024
done