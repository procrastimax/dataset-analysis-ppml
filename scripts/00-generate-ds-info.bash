#!/bin/bash
#SBATCH -J gen-ds-info
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='mnist fmnist cifar10 svhn'
        ;;
    1)
        ds='cifar10_gray svhn_gray'
        ;;
    2)
        ds='mnist_c5000 fmnist_c5000 cifar10_c5000 svhn_c5000'
        ;;
    3)
        ds='mnist_c4000 fmnist_c4000 cifar10_c4000 svhn_c4000'
        ;;
    4)
        ds='mnist_c3000 fmnist_c3000 cifar10_c3000 svhn_c3000'
        ;;
    5)
        ds='mnist_c2000 fmnist_c2000 cifar10_c2000 svhn_c2000'
        ;;
    6)
        ds='mnist_c1000 fmnist_c1000 cifar10_c1000 svhn_c1000'
        ;;
    7)
        ds='mnist_c500 fmnist_c500 cifar10_c500 svhn_c500'
        ;;
    8)
        ds='mnist_c100 fmnist_c100 cifar10_c100 svhn_c100'
        ;;
    9)
        ds='mnist_c50 fmnist_c50 cifar10_c50 svhn_c50'
        ;;
    10)
        ds='emnist cifar100'
        ;;
    11)
        ds='emnist_n47 cifar100_n47'
        ;;
    12)
        ds='emnist_n37 cifar100_n37'
        ;;
    13)
        ds='emnist_n27 cifar100_n27'
        ;;
    14)
        ds='emnist_n17 cifar100_n17'
        ;;
    15)
        ds='emnist_n7 cifar100_n7'
        ;;
    16)
        ds='emnist_n3 cifar100_n3'
        ;;
    17)
        ds='mnist_c5000_iL0.1 fmnist_c5000_iL0.1 cifar10_c5000_iL0.1 svhn_c5000_iL0.1'
        ;;
    18)
        ds='mnist_c5000_iL0.3 fmnist_c5000_iL0.3 cifar10_c5000_iL0.3 svhn_c5000_iL0.3'
        ;;
    19)
        ds='mnist_c5000_iL0.6 fmnist_c5000_iL0.6 cifar10_c5000_iL0.6 svhn_c5000_iL0.6'
        ;;
    20)
        ds='mnist_c5000_iL0.9 fmnist_c5000_iL0.9 cifar10_c5000_iL0.9 svhn_c5000_iL0.9'
        ;;
    21)
        ds='mnist_c5000_iN0.1 fmnist_c5000_iN0.1 cifar10_c5000_iN0.1 svhn_c5000_iN0.1'
        ;;
    22)
        ds='mnist_c5000_iN0.3 fmnist_c5000_iN0.3 cifar10_c5000_iN0.3 svhn_c5000_iN0.3'
        ;;
    23)
        ds='mnist_c5000_iN0.6 fmnist_c5000_iN0.6 cifar10_c5000_iN0.6 svhn_c5000_iN0.6'
        ;;
    24)
        ds='mnist_c5000_iN0.9 fmnist_c5000_iN0.9 cifar10_c5000_iN0.9 svhn_c5000_iN0.9'
        ;;
    25)
        ds='mnist_n9 fmnist_n9 cifar10_n9 svhn_n9'
        ;;
    26)
        ds='mnist_n8 fmnist_n8 cifar10_n8 svhn_n8'
        ;;
    27)
        ds='mnist_n7 fmnist_n7 cifar10_n7 svhn_n7'
        ;;
    28)
        ds='mnist_n6 fmnist_n6 cifar10_n6 svhn_n6'
        ;;
    29)
        ds='mnist_n5 fmnist_n5 cifar10_n5 svhn_n5'
        ;;
    30)
        ds='mnist_n4 fmnist_n4 cifar10_n4 svhn_n4'
        ;;
    31)
        ds='mnist_n3 fmnist_n3 cifar10_n3 svhn_n3'
        ;;
esac

srun singularity exec container-dataset-analysis.sif python3.9 src/main.py -d $ds --generate-ds-info --force-ds-info-regeneration
