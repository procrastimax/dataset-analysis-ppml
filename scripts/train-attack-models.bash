#!/bin/bash
#SBATCH -J non-private-attacker
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END

ds='mnist fmnist mnist_c5000 fmnist_c5000 cifar10 cifar10gray'
shadow_models=32
epochs=150
lr=0.001
l2_clip=1.0
microbatches=8

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                model="small_cnn"
                eps=0.0
                ;;
        1)
                model="private_small_cnn"
                eps=0.1
                ;;
        2)
                model="private_small_cnn"
                eps=1.0
                ;;
        3)
                model="private_small_cnn"
                eps=10
                ;;
        4)
                model="private_small_cnn"
                eps=50
                ;;
esac

echo "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"

module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
source env/bin/activate
srun python src/main.py -d $ds -m $model -s $shadow_models -r $SLURM_ARRAY_TASK_ID -l $lr --epochs $epochs -c $l2_clip -b $microbatches -e $eps --run-amia-attack --generate-results
