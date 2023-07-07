#!/bin/bash
#SBATCH -J non-private-attacker
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
epochs=50
model="small_cnn"
batch=200

echo "non-private attacking" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -s $shadow_models -r 0 --epochs $epochs --batch-size $batch --run-amia-attack --generate-results --force-model-retrain -n "non-private-attack-c5000"
