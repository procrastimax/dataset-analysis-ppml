#!/bin/bash
#SBATCH -J private-dataset-size-comparison
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

run_name="private-dataset-size-comparison"
model="private_cnn"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='mnist_c5000 cifar10_c5000 fmnist_c5000 svhn_c5000'
        eps=1
        ;;
    1)
        ds='mnist_c4000 cifar10_c4000 fmnist_c4000 svhn_c4000'
        eps=1
        ;;
    2)
        ds='mnist_c3000 cifar10_c3000 fmnist_c3000 svhn_c3000'
        eps=1
        ;;
    3)
        ds='mnist_c2000 cifar10_c2000 fmnist_c2000 svhn_c2000'
        eps=1
        ;;
    4)
        ds='mnist_c1000 cifar10_c1000 fmnist_c1000 svhn_c1000'
        eps=1
        ;;
    5)
        ds='mnist_c500 cifar10_c500 fmnist_c500 svhn_c500'
        eps=1
        ;;
    6)
        ds='mnist_c100 cifar10_c100 fmnist_c100 svhn_c100'
        eps=1
        ;;
    7)
        ds='mnist_c50 cifar10_c50 fmnist_c50 svhn_c50'
        eps=1
        ;;

    8)
        ds='mnist_c5000 cifar10_c5000 fmnist_c5000 svhn_c5000'
        eps=30
        ;;
    9)
        ds='mnist_c4000 cifar10_c4000 fmnist_c4000 svhn_c4000'
        eps=30
        ;;
    10)
        ds='mnist_c3000 cifar10_c3000 fmnist_c3000 svhn_c3000'
        eps=30
        ;;
    11)
        ds='mnist_c2000 cifar10_c2000 fmnist_c2000 svhn_c2000'
        eps=30
        ;;
    12)
        ds='mnist_c1000 cifar10_c1000 fmnist_c1000 svhn_c1000'
        eps=30
        ;;
    13)
        ds='mnist_c500 cifar10_c500 fmnist_c500 svhn_c500'
        eps=30
        ;;
    14)
        ds='mnist_c100 cifar10_c100 fmnist_c100 svhn_c100'
        eps=30
        ;;
    15)
        ds='mnist_c50 cifar10_c50 fmnist_c50 svhn_c50'
        eps=30
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name -e $eps
