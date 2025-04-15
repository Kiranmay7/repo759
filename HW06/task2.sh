#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J stencil
#SBATCH -o stencil_64.out
#SBATCH -e stencil-%j.err
hostname
for ((i=2**10; i<=2**29; i*=2))
do
  n=$i
  ./task2 $n 128 512
done