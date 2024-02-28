#!/bin/bash
#SBATCH --job-name=jzfov
#SBATCH --output=output_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2:59:00
#SBATCH --gres=gpu:v100:1

singularity exec --nv --overlay /scratch/jz5952/fov_env/my_pytorch.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh && cd /scratch/jz5952/FoV && python3 main.py --model MyTransformer  --num_epochs 50 --hist_time 2 --pred_time 2 --n_heads 3 --head_dim 5 --feature_names XYZ_FEATURE_NAMES"
# source /ext3/env.sh
# cd /scratch/jz5952/FoV

# # Run your Python code
# python3 main.py --model MyTransformer  --num_epochs 50 --hist_time 2 --pred_time 1
