#!/bin/bash
#SBATCH -J non-private-shadow-model-diff
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-15:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END

ds='mnist cifar10'
epochs=25
model="cnn"
batch=200

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                shadow_models=8
                ;;
        1)
                shadow_models=16
                ;;
        2)
                shadow_models=32
                ;;
        3)
                shadow_models=64
                ;;
esac

echo "private-attacking" "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -s $shadow_models -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch --run-amia-attack --include-mia --generate-results --force-model-retrain -n "non-private-shadow-model-diff"
