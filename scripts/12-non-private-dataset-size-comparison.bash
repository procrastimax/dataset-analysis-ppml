#!/bin/bash
#SBATCH -J non-private-dataset-size-comparison
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

run_name="non-private-dataset-size-comparison"
model="cnn"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='mnist_c5000 cifar10_c5000 fmnist_c5000 svhn_c5000'
        ;;
    1)
        ds='mnist_c4000 cifar10_c4000 fmnist_c4000 svhn_c4000'
        ;;
    2)
        ds='mnist_c3000 cifar10_c3000 fmnist_c3000 svhn_c3000'
        ;;
    3)
        ds='mnist_c2000 cifar10_c2000 fmnist_c2000 svhn_c2000'
        ;;
    4)
        ds='mnist_c1000 cifar10_c1000 fmnist_c1000 svhn_c1000'
        ;;
    5)
        ds='mnist_c500 cifar10_c500 fmnist_c500 svhn_c500'
        ;;
    6)
        ds='mnist_c100 cifar10_c100 fmnist_c100 svhn_c100'
        ;;
    7)
        ds='mnist_c50 cifar10_c50 fmnist_c50 svhn_c50'
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack -ca --force-model-retrain -n $run_name
