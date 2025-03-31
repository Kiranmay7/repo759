#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J factorial
#SBATCH -o factorial.out
#SBATCH -e factorial.err
hostname
./task1