#!/bin/bash
#SBATCH -J gen-ds-info
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=

# use: sbatch -a 1-4 large-attacker.job

ds='mnist fmnist mnist_c5000 fmnist_c5000 cifar10 cifar10gray'

echo "Generating ds-info for $ds"

module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

source env/bin/activate

srun python src/main.py -d $ds -r 0 --generate-ds-info
