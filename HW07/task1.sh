#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH -J matmul
#SBATCH -o matmul_c.out 
#SBATCH -e matmul.err
hostname
for ((i=16; i<=1024; i*=2))
do
  n=$i
  ./task1 8192 $n
done
