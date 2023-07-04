#!/bin/bash
#SBATCH -J private-attacker
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-15:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END

ds='mnist_c5000 fmnist_c5000 cifar10_c5000 cifar10gray_c5000'
shadow_models=32
epochs=15
model="private_small_cnn"
batch=200

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                eps=1.0
                ;;
        1)
                eps=30
                ;;
esac

echo "private-attacking" "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -e $eps -d $ds -m $model -s $shadow_models -r $SLURM_ARRAY_TASK_ID --epochs $epochs -b $batch --batch-size $batch --run-amia-attack --generate-results --force-model-retrain -n "private-attack-c5000"
