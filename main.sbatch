#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH --output=main_out.out

python3 main.py
