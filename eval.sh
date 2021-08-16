#!/bin/bash
#
# At first let's give job some descriptive name to distinct
# from other jobs running on cluster
#SBATCH -J JP_JOB
#
# Let's redirect job's out some other file than default slurm-%jobid-out
#SBATCH --output=res/res_eval_%a.txt
#SBATCH --error=res/err_eval_%a.txt
#
# We'll want to allocate one CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#
# We'll want to reserve 2GB memory for the job
# and 3 days of compute time to finish.
# Also define to use the GPU partition.
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --time=6-23:59:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=juanpablo.lagosbenitez@tuni.fi
#
# These commands will be executed on the compute node:

# Load all modules you need below. Edit these to your needs.

#module load matlab
#module lodad python
module load CUDA/9.0
# conda activate pynoptorch
source activate pynoptorch

# Finally run your job. Here's an example of a python script.

python eval_depth_completion.py $SLURM_TASK_ARRAY_ID