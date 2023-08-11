#!/bin/bash
#SBATCH -J non-private-class-imbalance
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

run_name="non-private-attack-class-imbalance"
model="cnn"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='mnist_c5000 cifar10_c5000 fmnist_c5000 svhn_c5000'
        eps=1
        ;;
    1)
        ds='mnist_c5000_iL0.1 cifar10_c5000_iL0.1 fmnist_c5000_iL0.1 svhn_c5000_iL0.1'
        eps=1
        ;;
    2)
        ds='mnist_c5000_iL0.3 cifar10_c5000_iL0.3 fmnist_c5000_iL0.3 svhn_c5000_iL0.3'
        eps=1
        ;;
    3)
        ds='mnist_c5000_iL0.6 cifar10_c5000_iL0.6 fmnist_c5000_iL0.6 svhn_c5000_iL0.6'
        eps=1
        ;;
    4)
        ds='mnist_c5000_iL0.9 cifar10_c5000_iL0.9 fmnist_c5000_iL0.9 svhn_c5000_iL0.9'
        eps=1
        ;;
    5)
        ds='mnist_c5000_iN0.1 cifar10_c5000_iN0.1 fmnist_c5000_iN0.1 svhn_c5000_iN0.1'
        eps=1
        ;;
    6)
        ds='mnist_c5000_iN0.3 cifar10_c5000_iN0.3 fmnist_c5000_iN0.3 svhn_c5000_iN0.3'
        eps=1
        ;;
    7)
        ds='mnist_c5000_iN0.6 cifar10_c5000_iN0.6 fmnist_c5000_iN0.6 svhn_c5000_iN0.6'
        eps=1
        ;;
    8)
        ds='mnist_c5000_iN0.9 cifar10_c5000_iN0.9 fmnist_c5000_iN0.9 svhn_c5000_iN0.9'
        eps=1
        ;;


    9)
        ds='mnist_c5000 cifar10_c5000 fmnist_c5000 svhn_c5000'
        eps=30
        ;;
    10)
        ds='mnist_c5000_iL0.1 cifar10_c5000_iL0.1 fmnist_c5000_iL0.1 svhn_c5000_iL0.1'
        eps=30
        ;;
    11)
        ds='mnist_c5000_iL0.3 cifar10_c5000_iL0.3 fmnist_c5000_iL0.3 svhn_c5000_iL0.3'
        eps=30
        ;;
    12)
        ds='mnist_c5000_iL0.6 cifar10_c5000_iL0.6 fmnist_c5000_iL0.6 svhn_c5000_iL0.6'
        eps=30
        ;;
    13)
        ds='mnist_c5000_iL0.9 cifar10_c5000_iL0.9 fmnist_c5000_iL0.9 svhn_c5000_iL0.9'
        eps=30
        ;;
    14)
        ds='mnist_c5000_iN0.1 cifar10_c5000_iN0.1 fmnist_c5000_iN0.1 svhn_c5000_iN0.1'
        eps=30
        ;;
    15)
        ds='mnist_c5000_iN0.3 cifar10_c5000_iN0.3 fmnist_c5000_iN0.3 svhn_c5000_iN0.3'
        eps=30
        ;;
    16)
        ds='mnist_c5000_iN0.6 cifar10_c5000_iN0.6 fmnist_c5000_iN0.6 svhn_c5000_iN0.6'
        eps=30
        ;;
    17)
        ds='mnist_c5000_iN0.9 cifar10_c5000_iN0.9 fmnist_c5000_iN0.9 svhn_c5000_iN0.9'
        eps=30
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name -e $eps
