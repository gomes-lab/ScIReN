#!/bin/bash

# Runs BINN training on GPUs, as a Slurm job. Usage:
# sbatch run_kan_onelayer.sh
# (To change the number of GPUs, change --gpus and --num_CPU arguments to that number.)
# (To run on CPU, remove the --gpus line and set --num_CPU to the number of CPUs.)
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the full partition, which contains the GPU nodes.
#SBATCH -p regular
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J kan_onelayer
# Request 4 GPUs 
# #SBATCH --gpus 4
# Request 4 CPU cores (9 hyperthreads).
#SBATCH -c 8
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 50GB RAM
#SBATCH --mem=80GB
# Request a walltime limit of 72 hours
#SBATCH -t 72:00:00

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


# Set number of threads
# export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/2))

# # Use scratch for temporary files to avoid space limits in /tmp
# export TMPDIR=/glade/scratch/$USER/temp
# mkdir -p $TMPDIR

# Load modules to match compile-time environment
# module purge
source ~/.bashrc
# module load cuda
# module load mkl

# Activate environment
source .venv/bin/activate


for LR in 1e-1 1e-2
do
    for LAM in 1
    do
        for FOLD in 1 2 3 4 5
        do
            SEED=$FOLD
            python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --lr $LR --optimizer AdamW --batch_size 32 \
                --seed $SEED --init default --min_temp 1 --max_temp 1 \
                --n_epochs 200 --patience 100 --model kan \
                --num_layers 1 --features ten \
                --losses smooth_l1 param_reg param_violation kan_l1 kan_entropy kan_coefdiff --lambdas 1 1 1000 1 2 0 \
                --param_constraint hardsigmoid \
                --kan_grid 6 --kan_update_grid 1 --kan_grid_margin 1.0 --kan_base_fun identity --kan_affine_trainable \
                --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "KAN_ONELAYER_GRIDMARGIN" --plot
        done
    done
done

# --kan_update_grid 1 --kan_affine_trainable   --jacobian_noise_std 1.0
# \ --loss_weighting two_stage --second_start 50 --second_lambdas 1 1 100 1 100 \

# for LR in 1e-3 1e-2 1e-1
# do
#     for PREG in 1 10 100
#     do
#         for FOLD in 1 2 3 4 5
#         do
#             SEED=$FOLD
#             python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
#                 --lr $LR --optimizer AdamW \
#                 --seed $SEED --init default --min_temp 1 --max_temp 1 \
#                 --n_epochs 200 --patience 100 --model kan \
#                 --num_layers 1 --features ten --standardize_input \
#                 --losses smooth_l1 unconstrained_param --lambdas 1 $PREG --param_constraint hardsigmoid \
#                 --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "KAN_ONELAYER_STD" --plot
#         done
#     done
# done
# --loss_weighting two_stage --second_start 100 --second_lambdas 1 1 1 100 \


# # Hardsigmoid tuning
# for LR in 1e-4 1e-3 1e-2
# do
#     for TEMP in 0.5 1 3
#     do
#         for PREG in 0 10 100
#         do
#             for FOLD in 1
#             do
#                 if [ $FOLD -eq 1 ]; then
#                     PLOT_STR="--plot"
#                 else
#                     PLOT_STR=""
#                 fi

#                 SEED=$FOLD
#                 python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
#                     --lr $LR --optimizer AdamW --weight_decay 0 \
#                     --seed $SEED --init default --min_temp $TEMP --max_temp $TEMP \
#                     --n_epochs 200 --patience 100 --model new_mlp  \
#                     --num_layers 3 --residual --width 256 \
#                     --activation leaky_relu --use_bn --categorical one_hot --pos_enc none \
#                     --losses smooth_l1 param_reg param_violation --lambdas 1 $PREG 1000 --param_constraint hardsigmoid \
#                     --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 71.5 --note "BINN_HARDSIGMOID_TUNING" $PLOT_STR
#             done
#         done
#     done
# done

