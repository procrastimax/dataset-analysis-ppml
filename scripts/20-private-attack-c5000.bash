#!/bin/bash
#SBATCH -J private-attack-c5000
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist_c5000 fmnist_c5000 cifar10_c5000 cifar10_c5000_gray svhn_c5000 svhn_c5000_gray'
shadow_models=32
epochs=10
model="private_cnn"
batch=200
run_name="private-attack-c5000"

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                eps=1
                ;;
        1)
                eps=30
                ;;
esac

echo $run_name "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -e $eps -d $ds -m $model -s $shadow_models -r $SLURM_ARRAY_TASK_ID --epochs $epochs -b $batch --batch-size $batch --run-amia-attack --generate-results --force-model-retrain -n $run_name
