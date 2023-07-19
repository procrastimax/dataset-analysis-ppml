#!/bin/bash
#SBATCH -J non-private-class-imbalance
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-15:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END

run_name="non-private-attack-class-imbalance"
shadow_models=32
epochs=50
model="cnn"
batch=200

case $SLURM_ARRAY_TASK_ID in
        0)
            ds='mnist_c5000 cifar10_c5000'
                ;;
        1)
            ds='mnist_c5000_iN0.75 cifar10_c5000_iN0.75'
                ;;
        2)
            ds='mnist_c5000_iN0.5 cifar10_c5000_iN0.5'
                ;;
        3)
            ds='mnist_c5000_iN0.25 cifar10_c5000_iN0.25'
                ;;
        4)
            ds='mnist_c5000_iL0.75 cifar10_c5000_iL0.75'
                ;;
        5)
            ds='mnist_c5000_iL0.5 cifar10_c5000_iL0.5'
                ;;
        6)
            ds='mnist_c5000_iL0.25 cifar10_c5000_iL0.25'
                ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -s $shadow_models -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch --run-amia-attack --generate-results --force-model-retrain -n $run_name
