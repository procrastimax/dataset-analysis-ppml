#!/bin/bash
#SBATCH -J private-attacker-class-imbalance
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-15:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END

run_name="private-attack-class-imbalance"
shadow_models=32
epochs=10
model="private_small_cnn"
batch=200

case $SLURM_ARRAY_TASK_ID in
        0)
            ds='mnist cifar10'
            eps=0.1
                ;;
        1)
            ds='mnist cifar10'
            eps=30
                ;;
        2)
            ds='mnist_iN0.75 cifar10_iN0.75'
            eps=0.1
                ;;
        3)
            ds='mnist_iN0.5 cifar10_iN0.5'
            eps=0.1
                ;;
        4)
            ds='mnist_iN0.25 cifar10_iN0.25'
            eps=0.1
                ;;
        5)
            ds='mnist_iL0.75 cifar10_iL0.75'
            eps=0.1
                ;;
        6)
            ds='mnist_iL0.5 cifar10_iL0.5'
            eps=0.1
                ;;
        7)
            ds='mnist_iL0.25 cifar10_iL0.25'
            eps=0.1
                ;;
        8)
            ds='mnist_iN0.75 cifar10_iN0.75'
            eps=30
                ;;
        9)
            ds='mnist_iN0.5 cifar10_iN0.5'
            eps=30
                ;;
        10)
            ds='mnist_iN0.25 cifar10_iN0.25'
            eps=30
                ;;
        11)
            ds='mnist_iL0.75 cifar10_iL0.75'
            eps=30
                ;;
        12)
            ds='mnist_iL0.5 cifar10_iL0.5'
            eps=30
                ;;
        13)
            ds='mnist_iL0.25 cifar10_iL0.25'
            eps=30
                ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -s $shadow_models -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch -e $eps --run-amia-attack --generate-results --force-model-retrain -n $run_name
