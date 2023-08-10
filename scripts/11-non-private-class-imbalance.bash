#!/bin/bash
#SBATCH -J non-private-class-imbalance
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

run_name="non-private-attack-class-imbalance"
model="cnn"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='mnist_c5000 cifar10_c5000 fmnist_c5000 svhn_c5000'
        ;;
    1)
        ds='mnist_c5000_iN0.75 cifar10_c5000_iN0.75 fmnist_c5000_iN0.75 svhn_c5000_iN0.75'
        ;;
    2)
        ds='mnist_c5000_iN0.5 cifar10_c5000_iN0.5 fmnist_c5000_iN0.5 svhn_c5000_iN0.5'
        ;;
    3)
        ds='mnist_c5000_iN0.25 cifar10_c5000_iN0.25 fmnist_c5000_iN0.25 svhn_c5000_iN0.25'
        ;;
    4)
        ds='mnist_c5000_iL0.75 cifar10_c5000_iL0.75 fmnist_c5000_iL0.75 svhn_c5000_iL0.75'
        ;;
    5)
        ds='mnist_c5000_iL0.5 cifar10_c5000_iL0.5 fmnist_c5000_iL0.5 svhn_c5000_iL0.5'
        ;;
    6)
        ds='mnist_c5000_iL0.25 cifar10_c5000_iL0.25 fmnist_c5000_iL0.25 svhn_c5000_iL0.25'
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack -ca --force-model-retrain -n $run_name
