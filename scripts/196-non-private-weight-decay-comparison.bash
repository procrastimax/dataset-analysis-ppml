#!/bin/bash
#SBATCH -J non-private-weight-decay-comparison
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-5:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist fmnist svhn cifar10'
model="cnn"
batch=600
run_name="non-private-weight-decay-comparison"
epochs=30
lr=0.005

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                wd=0
                ;;
        1)
                wd=0.0001
                ;;
        2)
                wd=0.0005
                ;;
        3)
                wd=0.001
                ;;
        4)
                wd=0.005
                ;;
        5)
                wd=0.01
                ;;
        6)
                wd=0.05
                ;;
        7)
                wd=0.1
                ;;
        8)
                wd=0.2
                ;;
esac

echo "non-private" "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch -n $run_name -tm -em -wd $wd -l $lr
