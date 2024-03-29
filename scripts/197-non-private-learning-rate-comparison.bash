#!/bin/bash
#SBATCH -J learning-rate-comparison
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-0:30:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist fmnist svhn cifar10'
model="cnn"
batch=256
run_name="learning-rate-comparison"
epochs=30

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                lr=0.0001
                ;;
        1)
                lr=0.001
                ;;
        2)
                lr=0.005
                ;;
        3)
                lr=0.01
                ;;
        4)
                lr=0.05
                ;;
        5)
                lr=0.1
                ;;
        6)
                lr=1.0
                ;;
esac

echo "non-private" "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch -n $run_name -tm -em -l $lr
