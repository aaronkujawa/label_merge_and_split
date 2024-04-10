#!/bin/bash
#SBATCH --partition compute           # Submit to 'compute' Partition or queue
#SBATCH --job-name=multiatlasprop
#SBATCH --output=logs/run_multiatlasprop_array_%A_%a.out
#SBATCH -e logs/run_multiatlasprop_array_%A_%a.err
#SBATCH --time=0-72:00:00        # Run for a maximum time of 0 days, 72 hours, 00 mins, 00 secs
#SBATCH --nodes=1            # Request N nodes
#SBATCH --ntasks-per-node=64  # Request n cores or task per node
#SBATCH --mem-per-cpu=3500M

# the environment variable SLURM_ARRAY_TASK_ID contains the index corresponding to the current job step

#echo ------ $1
#./runGIF_array_program.sh "$@"

# script path input
script_path=${1}
nb_of_submitted_jobs_limit=${2}

module load miniconda
conda activate local

echo Submitting ${script_path} --taskid ${SLURM_ARRAY_TASK_ID}

timestamp=$(date -d "today" +"%Y-%m-%d-%H_%M_%S")
#python -m cProfile -o logs/profile_${timestamp}.dat ${script_path}
python ${script_path} --taskid ${SLURM_ARRAY_TASK_ID}

#if [[ $(((SLURM_ARRAY_TASK_ID+1) % nb_of_submitted_jobs_limit)) == 0 ]] ; then 
#    sbatch --array=$((SLURM_ARRAY_TASK_ID+1))-$((SLURM_ARRAY_TASK_ID+nb_of_submitted_jobs_limit)) "$0" "$@"
#fi

# nb_cases=${#imag_paths[@]}
# next_task_id=$((SLURM_ARRAY_TASK_ID+nb_of_submitted_jobs_limit))
# # submit job from next batch of array jobs
# if (( next_task_id < nb_cases )); then
# 	sbatch --array=$next_task_id "$0" "$@"
# fi
