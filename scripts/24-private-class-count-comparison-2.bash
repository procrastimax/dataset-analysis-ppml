#!/bin/bash
#SBATCH -J private-class-count-comparison-2
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

run_name="private-class-count-comparison-2"
model="private_cnn"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='mnist_c10 fmnist_c10 svhn_c10 cifar10_c10'
        eps=1
        ;;
    1)
        ds='mnist_c9 fmnist_c9 svhn_c9 cifar10_c9'
        eps=1
        ;;
    2)
        ds='mnist_c8 fmnist_c8 svhn_c8 cifar10_c8'
        eps=1
        ;;
    3)
        ds='mnist_c7 fmnist_c7 svhn_c7 cifar10_c7'
        eps=1
        ;;
    4)
        ds='mnist_c6 fmnist_c6 svhn_c6 cifar10_c6'
        eps=1
        ;;
    5)
        ds='mnist_c5 fmnist_c5 svhn_c5 cifar10_c5'
        eps=1
        ;;
    6)
        ds='mnist_c4 fmnist_c4 svhn_c4 cifar10_c4'
        eps=1
        ;;
    7)
        ds='mnist_c3 fmnist_c3 svhn_c3 cifar10_c3'
        eps=1
        ;;
    8)
        ds='mnist_c10 fmnist_c10 svhn_c10 cifar10_c10'
        eps=30
        ;;
    9)
        ds='mnist_c9 fmnist_c9 svhn_c9 cifar10_c9'
        eps=30
        ;;
    10)
        ds='mnist_c8 fmnist_c8 svhn_c8 cifar10_c8'
        eps=30
        ;;
    11)
        ds='mnist_c7 fmnist_c7 svhn_c7 cifar10_c7'
        eps=30
        ;;
    12)
        ds='mnist_c6 fmnist_c6 svhn_c6 cifar10_c6'
        eps=30
        ;;
    13)
        ds='mnist_c5 fmnist_c5 svhn_c5 cifar10_c5'
        eps=30
        ;;
    14)
        ds='mnist_c4 fmnist_c4 svhn_c4 cifar10_c4'
        eps=30
        ;;
    15)
        ds='mnist_c3 fmnist_c3 svhn_c3 cifar10_c3'
        eps=30
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name -e $eps
