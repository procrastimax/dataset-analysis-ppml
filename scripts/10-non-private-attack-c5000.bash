#!/bin/bash
#SBATCH -J non-private-attack-c5000
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist_c5000 fmnist_c5000 cifar10_c5000 cifar10_c5000_gray svhn_c5000 svhn_c5000_gray'
shadow_models=32
epochs=50
model="cnn"
batch=200
run_name="non-private-attack-c5000"

echo $run_name "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -s $shadow_models -r 0 --epochs $epochs --batch-size $batch --run-amia-attack --generate-results --force-model-retrain -n $run_name
