#!/bin/bash
#SBATCH -J private-attacker-class-imbalance
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

run_name="private-attack-class-imbalance"
model="private_cnn"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='mnist_c5000 cifar10_c5000 svhn_c5000 fmnist_c5000'
        eps=1
        ;;

    1)
        ds='mnist_c5000_iN0.75 cifar10_c5000_iN0.75 svhn_c5000_iN0.75 fmnist_c5000_iN0.75'
        eps=1
        ;;

    2)
        ds='mnist_c5000_iN0.5 cifar10_c5000_iN0.5 svhn_c5000_iN0.5 fmnist_c5000_iN0.5'
        eps=1
        ;;
    3)
        ds='mnist_c5000_iN0.25 cifar10_c5000_iN0.25 svhn_c5000_iN0.25 fmnist_c5000_iN0.25'
        eps=1
        ;;


    4)
        ds='mnist_c5000_iL0.75 cifar10_c5000_iL0.75 svhn_c5000_iL0.75 fmnist_c5000_iL0.75'
        eps=1
        ;;
    5)
        ds='mnist_c5000_iL0.5 cifar10_c5000_iL0.5 svhn_c5000_iL0.5 fmnist_c5000_iL0.5'
        eps=1
        ;;
    6)
        ds='mnist_c5000_iL0.25 cifar10_c5000_iL0.25 svhn_c5000_iL0.25 fmnist_c5000_iL0.25'
        eps=1
        ;;




    7)
        ds='mnist_c5000 cifar10_c5000 svhn_c5000 fmnist_c5000'
        eps=30
        ;;

    8)
        ds='mnist_c5000_iN0.75 cifar10_c5000_iN0.75 svhn_c5000_iN0.75 fmnist_c5000_iN0.75'
        eps=30
        ;;
    9)
        ds='mnist_c5000_iN0.5 cifar10_c5000_iN0.5 svhn_c5000_iN0.5 fmnist_c5000_iN0.5'
        eps=30
        ;;
    10)
        ds='mnist_c5000_iN0.25 cifar10_c5000_iN0.25 svhn_c5000_iN0.25 fmnist_c5000_iN0.25'
        eps=30
        ;;


    11)
        ds='mnist_c5000_iL0.75 cifar10_c5000_iL0.75 svhn_c5000_iL0.75 fmnist_c5000_iL0.75'
        eps=30
        ;;
    12)
        ds='mnist_c5000_iL0.5 cifar10_c5000_iL0.5 svhn_c5000_iL0.5 fmnist_c5000_iL0.5'
        eps=30
        ;;
    13)
        ds='mnist_c5000_iL0.25 cifar10_c5000_iL0.25 svhn_c5000_iL0.25 fmnist_c5000_iL0.25'
        eps=30
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID -e $eps --run-amia-attack -ca --force-model-retrain -n $run_name
