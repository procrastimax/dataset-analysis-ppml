#!/bin/bash
#SBATCH -J non-private-ema-momentum-comparison
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
run_name="non-private-ema-momentum-comparison"
epochs=30

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                ema_momentum=0.0
                ;;
        1)
                ema_momentum=0.1
                ;;
        2)
                ema_momentum=0.333
                ;;
        3)
                ema_momentum=0.666
                ;;
        4)
                ema_momentum=0.999
                ;;
esac

echo "non-private" "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch -n $run_name -tm -ema $ema_momentum
