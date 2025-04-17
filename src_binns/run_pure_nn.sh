#!/bin/bash

# Runs pure-NN training. Usage:
# sbatch run_pure_nn.sh
# (To change the number of CPUs, modify the --num_CPU argument.)
# (To use GPUs, uncomment the '#SBATCH --gpus' line, change it to the number of GPUs, and set --num_CPU to that number
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Alternatively, if you need to run this interactively:
# 1) Request job. On slurm, an example command to request 8 CPUs is:
# srun -p regular -n 1 -c 8 --time=120:00:00 --mem-per-cpu=10G --pty /bin/bash -l
# 2) Run the script. (--num_CPU should be set to number of GPUs if available, otherwise the number of CPUs.)
# ./run_pure_nn.sh

# ??? TODO Figure out what is wrong with the vertical mixing. Right now setting simple_two_intercepts only for the "bulk simulations" which are actually meaningless.

# ================================== SLURM BOILERPLATE ======================================
# -p specifies the partition name. On AIDA cluster, use "-p full" if using GPU; otherwise use "-p regular".
#SBATCH -p regular
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J pure_nn
# If you wanted to request GPUs, uncomment out this line.
# #SBATCH --gpus 4
# Request 4 CPU cores (8 hyperthreads).
#SBATCH -c 8
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Amount of RAM needed
#SBATCH --mem=80GB
# Walltime limit (72 hours)
#SBATCH -t 72:00:00


# Load modules to match compile-time environment
# module purge
source ~/.bashrc
# module load cuda
# module load mkl

# Activate environment
source .venv/bin/activate


# TUNING: Pure NN, Smooth L1 loss, ten features
for LR in 1e-4 1e-3 1e-2 1e-1
do
    for WD in 0 1e-4 1e-3
    do
        for FOLD in 1 2 3 4 5
        do
            SEED=$FOLD
            python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --optimizer AdamW --lr $LR --weight_decay $WD --seed $SEED \
                --features ten --standardize_output \
                --n_epochs 200 --patience 100 --model nn_only --vertical_mixing original \
                --num_layers 3 --residual --activation leaky_relu --use_bn \
                --losses smooth_l1 --lambdas 1 \
                --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "NNONLY_TEN"
        done
    done
done



# # TUNING: STANDARDIZE OUTPUT+INPUT
# for LR in 1e-4 1e-3 1e-2 1e-1 1 10 100
# do
#     for FOLD in 1
#     do
#         for SEED in 0 1 2
#         do
#             python3 binns_DDP.py --data_seed 12345 --n_datapoints 400 --split vertical --cross_val_idx $FOLD --n_folds 5 \
#                 --lr $LR --optimizer AdamW --weight_decay 0 \
#                 --seed $SEED --init default --standardize_output --standardize_input \
#                 --n_epochs 200 --patience 100 --model nn_only --vertical_mixing simple_two_intercepts \
#                 --activation leaky_relu --use_bn --embed_dim 5 --pos_enc none \
#                 --losses l1 --loss_weighting manual --lambdas 1 \
#                 --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "NNONLY_STDOUTPUTINPUT_VERTICAL400_TUNING"
#         done
#     done
# done


# # TUNING: NO STANDARDIZE OUTPUT
# for LR in 1e-4 1e-3 1e-2 1e-1 1 10 100
# do
#     for FOLD in 1
#     do
#         for SEED in 0 1 2
#         do
#             python3 binns_DDP.py --data_seed 12345 --n_datapoints 200 --split horizontal --cross_val_idx $FOLD --n_folds 5 \
#                 --lr $LR --optimizer AdamW --weight_decay 0 \
#                 --seed $SEED --init default \
#                 --n_epochs 200 --patience 100 --model nn_only --vertical_mixing simple_two_intercepts \
#                 --activation leaky_relu --use_bn --embed_dim 5 --pos_enc none \
#                 --losses l1 --loss_weighting manual --lambdas 1 \
#                 --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "NNONLY_TUNING"
#         done
#     done
# done

# # TUNING: STANDARDIZE OUTPUT
# for LR in 1e-4 1e-3 1e-2 1e-1 1 10 100
# do
#     for FOLD in 1
#     do
#         for SEED in 0 1 2
#         do
#             python3 binns_DDP.py --data_seed 12345 --n_datapoints 200 --split horizontal --cross_val_idx $FOLD --n_folds 5 \
#                 --lr $LR --optimizer AdamW --weight_decay 0 \
#                 --seed $SEED --init default --standardize_output \
#                 --n_epochs 200 --patience 100 --model nn_only --vertical_mixing simple_two_intercepts \
#                 --activation leaky_relu --use_bn --embed_dim 5 --pos_enc none \
#                 --losses l1 --loss_weighting manual --lambdas 1 \
#                 --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "NNONLY_TUNING_STDOUTPUT"
#         done
#     done
# done

# for LR in 1e-2
# do
#     for FOLD in 1 2 3 4 5
#     do
#         for SEED in 0 1 2
#         do
#             python3 binns_DDP.py --data_seed 12345 --n_datapoints 200 --split horizontal --cross_val_idx $FOLD --n_folds 5 \
#                 --lr $LR --optimizer AdamW --weight_decay 0 --seed $SEED \
#                 --n_epochs 200 --patience 100 --model nn_only --vertical_mixing simple_two_intercepts \
#                 --activation leaky_relu --use_bn --embed_dim 5 --pos_enc none \
#                 --losses l1 --loss_weighting manual --lambdas 1 \
#                 --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "NNONLY"
#             exit
#         done
#     done
# done
