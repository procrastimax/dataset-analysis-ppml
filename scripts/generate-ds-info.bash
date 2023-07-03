#!/bin/bash
#SBATCH -J gen-ds-info
#SBATCH --ntasks=1
#SBATCH --mem=25G
#SBATCH --partition=clara
#SBATCH --time=0-10:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=

ds='mnist_c5000 fmnist_c5000 cifar10_c5000 cifar10gray_c5000'
echo "Generating ds-info for $ds"
srun singularity exec container-dataset-analysis.sif python3.9 src/main.py -d $ds --generate-ds-info
