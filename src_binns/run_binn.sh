#!/bin/bash

# Runs BINN training. You can either run this script interactively or as a slurm job.
# 1) Interactively:
# Request job. On slurm, an example command to request 8 CPUs is
# srun -p full -n 1 -c 8 --time=72:00:00 --mem-per-cpu=10G --pty /bin/bash -l
# Run the script. From src_binns directory:
# ./run_binn.sh
#
# 2) Via SLURM: Simply run
# sbatch run_binn.sh


# ================================== SLURM BOILERPLATE ======================================
# On aida, the full partition contains GPU nodes, while the regular partition only contains CPU.
# For now CPU seems faster than GPU so use CPU.
#SBATCH -p regular
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J binn_std
# If you wanted to request GPUs, uncomment out this line.
# #SBATCH --gpus 4
# Request 16 CPU cores (32 hyperthreads).
#SBATCH -c 64
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 100GB RAM
#SBATCH --mem=200GB
# Request a walltime limit of 72 hours
#SBATCH -t 120:00:00

# INFO: Print properties of job as submitted
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NAME = $SLURM_JOB_NAME"
echo "SLURM_JOB_PARTITION = $SLURM_JOB_PARTITION"
echo "SLURM_SUBMIT_HOST = $SLURM_SUBMIT_HOST"
echo "SLURM_NTASKS = $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE = $SLURM_MEM_PER_NODE"
echo "SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES"
echo "SLURM_GPUS = $SLURM_GPUS"

# INFO: Print properties of job as scheduled by Slurm
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "SLURMD_NODENAME = $SLURMD_NODENAME"
echo "SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE = $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# Load modules to match compile-time environment
# module purge
source ~/.bashrc
# module load cuda
# module load mkl

# Activate environment
source .venv/bin/activate

# # Single run. NOTES:
# # --representative sample restricts to 1000 representative sites. Remove to use the whole dataset (~26000 sites)
# # --losses can be any number of losses, see its help. You can weight them manually using --lambdas.
# # --num_CPU should be set to number of GPUs if GPUs are available, otherwise the number of CPUs.
# # --note is a string that describes this run; it's used in the output directory name, and we append the final
# # metrics to a result excel file with that name.
# # See documentation in binns_DDP for info on other arguments.
# for LR in 1e-3
# do
#     for FOLD in 1
#     do
#         for SEED in 0
#         do
#             python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
#                 --lr $LR --optimizer AdamW --weight_decay 0 \
#                 --seed $SEED --init default --min_temp 1 --max_temp 1 \
#                 --n_epochs 200 --patience 100 --model new_mlp --vertical_mixing original --vectorized yes \
#                 --activation leaky_relu --use_bn --categorical one_hot --pos_enc none \
#                 --losses smooth_l1 param_reg --loss_weighting manual --lambdas 1 100 \
#                 --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "DEBUG"  # "BINN_EXAMPLE" --plot
#         done
#     done
# done
# exit


# # Reproducing the BINN paper
# for LR in 1e-2
# do
#     for FOLD in 1 2 3 4 5 6 7 8 9 10
#     do
#         for SEED in 0
#         do
#             python3 binns_DDP.py --data_seed 12345 --split random --cross_val_idx $FOLD --n_folds 10 \
#                 --optimizer AdamW --lr $LR --weight_decay 0 \
#                 --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
#                 --n_epochs 50 --patience 10 --model new_mlp --vertical_mixing original --vectorized yes \
#                 --activation leaky_relu --use_bn --embed_dim 5 --pos_enc early \
#                 --losses smooth_l1 param_reg --lambdas 1 100 \
#                 --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "REPRO_BINN"
#         done
#     done
# done

# Newer experiment setup
for LR in 1e-3 1e-2
do
    for TEMP in 1
    do
        for FOLD in 1 2 3 4 5
        do
            SEED=$FOLD
            python3 binns_DDP.py --data_seed 12345 --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --optimizer AdamW --lr $LR --weight_decay 0 \
                --seed $SEED --init default --min_temp $TEMP --max_temp $TEMP \
                --n_epochs 100 --patience 20 --model new_mlp \
                --num_layers 3 --residual --width 256 \
                --activation leaky_relu --use_bn --embed_dim 5 --pos_enc none \
                --losses smooth_l1 param_reg --lambdas 1 100 \
                --num_CPU 32 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "BINN_STD"
        done
    done
done

# To resume from a previous partial run, run something like this.
# Note that "--whether_resume" should be set to 1, and --previous_job_id" should be set to the output folder name we load from.
# for LR in 1e-2
# do
#     for FOLD in 1
#     do
#         for SEED in 0
#         do
#             python3 binns_DDP.py --data_seed 12345 --split random --cross_val_idx $FOLD --n_folds 10 \
#                 --optimizer AdamW --lr $LR --weight_decay 0 \
#                 --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
#                 --n_epochs 200 --patience 100 --model new_mlp --vertical_mixing original \
#                 --activation leaky_relu --use_bn --embed_dim 5 --pos_enc early \
#                 --losses smooth_l1 param_reg --lambdas 1 100 \
#                 --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "REPRO_BINN" \
#                 --whether_resume 1 --previous_job_id "20250127-214804_BINN_REPRO_lr=1e-02_seed=0_fold=1"
#         done
#     done
# done

