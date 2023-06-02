#!/bin/bash
#SBATCH -J large-amia-attacker
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00:00
#SBATCH -o "$HOME"/logs/%A-%x-%a.out
#SBATCH -e "$HOME"/logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=

# use: sbatch -a 1-4 large-attacker.job

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
  0)
    ds='mnist fmnist mnist_c5000 fmnist_c5000 cifar10 cifar10gray'
    shadow_models='16'
    ;;
  1)
    ds='mnist cifar10'
    shadow_models='16'
    ;;
  2)
    ds='mnist cifar10'
    shadow_models='32'
    ;;
  3)
    ds='mnist cifar10'
    shadow_models='64'
    ;;
  4)
    ds='mnist cifar10'
    shadow_models='128'
    ;;
esac

echo "$SLURM_ARRAY_TASK_ID" "-" "$ds" "-" "$shadow_models"

module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

source env/bin/activate

srun python src/main.py -d "$ds" -s "$shadow_models" -r "$SLURM_ARRAY_TASK_ID" --run-amia-attack
