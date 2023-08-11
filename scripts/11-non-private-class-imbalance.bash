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
        ds='mnist_c5000_iL0.1 cifar10_c5000_iL0.1 fmnist_c5000_iL0.1 svhn_c5000_iL0.1'
        ;;
    2)
        ds='mnist_c5000_iL0.3 cifar10_c5000_iL0.3 fmnist_c5000_iL0.3 svhn_c5000_iL0.3'
        ;;
    3)
        ds='mnist_c5000_iL0.6 cifar10_c5000_iL0.6 fmnist_c5000_iL0.6 svhn_c5000_iL0.6'
        ;;
    4)
        ds='mnist_c5000_iL0.9 cifar10_c5000_iL0.9 fmnist_c5000_iL0.9 svhn_c5000_iL0.9'
        ;;
    5)
        ds='mnist_c5000_iN0.1 cifar10_c5000_iN0.1 fmnist_c5000_iN0.1 svhn_c5000_iN0.1'
        ;;
    6)
        ds='mnist_c5000_iN0.3 cifar10_c5000_iN0.3 fmnist_c5000_iN0.3 svhn_c5000_iN0.3'
        ;;
    7)
        ds='mnist_c5000_iN0.6 cifar10_c5000_iN0.6 fmnist_c5000_iN0.6 svhn_c5000_iN0.6'
        ;;
    8)
        ds='mnist_c5000_iN0.9 cifar10_c5000_iN0.9 fmnist_c5000_iN0.9 svhn_c5000_iN0.9'
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name
