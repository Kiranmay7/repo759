#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J vscale
#SBATCH -o vscale.out
#SBATCH -e vscale-%j.err
hostname
for ((i=2**10; i<=2**29; i*=2))
do
  n=$i
  ./task3 $n
done