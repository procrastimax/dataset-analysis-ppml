#!/bin/bash
#SBATCH -J non-private-attack-c5000
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-10:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist_c5000 fmnist_c5000 cifar10_c5000 cifar10_c5000_gray svhn_c5000 svhn_c5000_gray'
model="cnn"
run_name="non-private-attack-c5000"

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r 0 --run-amia-attack --force-model-retrain -n $run_name
