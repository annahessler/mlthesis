#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=augment_output.out

python3 augment.py
