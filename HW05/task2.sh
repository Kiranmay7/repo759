#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J saxpy
#SBATCH -o saxpy.out
#SBATCH -e saxpy.err
hostname
./task2