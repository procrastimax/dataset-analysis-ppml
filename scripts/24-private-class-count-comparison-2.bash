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
        ds='mnist_n10 fmnist_n10 svhn_n10 cifar10_n10'
        eps=1
        ;;
    1)
        ds='mnist_n9 fmnist_n9 svhn_n9 cifar10_n9'
        eps=1
        ;;
    2)
        ds='mnist_n8 fmnist_n8 svhn_n8 cifar10_n8'
        eps=1
        ;;
    3)
        ds='mnist_n7 fmnist_n7 svhn_n7 cifar10_n7'
        eps=1
        ;;
    4)
        ds='mnist_n6 fmnist_n6 svhn_n6 cifar10_n6'
        eps=1
        ;;
    5)
        ds='mnist_n5 fmnist_n5 svhn_n5 cifar10_n5'
        eps=1
        ;;
    6)
        ds='mnist_n4 fmnist_n4 svhn_n4 cifar10_n4'
        eps=1
        ;;
    7)
        ds='mnist_n3 fmnist_n3 svhn_n3 cifar10_n3'
        eps=1
        ;;
    8)
        ds='mnist_n10 fmnist_n10 svhn_n10 cifar10_n10'
        eps=30
        ;;
    9)
        ds='mnist_n9 fmnist_n9 svhn_n9 cifar10_n9'
        eps=30
        ;;
    10)
        ds='mnist_n8 fmnist_n8 svhn_n8 cifar10_n8'
        eps=30
        ;;
    11)
        ds='mnist_n7 fmnist_n7 svhn_n7 cifar10_n7'
        eps=30
        ;;
    12)
        ds='mnist_n6 fmnist_n6 svhn_n6 cifar10_n6'
        eps=30
        ;;
    13)
        ds='mnist_n5 fmnist_n5 svhn_n5 cifar10_n5'
        eps=30
        ;;
    14)
        ds='mnist_n4 fmnist_n4 svhn_n4 cifar10_n4'
        eps=30
        ;;
    15)
        ds='mnist_n3 fmnist_n3 svhn_n3 cifar10_n3'
        eps=30
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name -e $eps
