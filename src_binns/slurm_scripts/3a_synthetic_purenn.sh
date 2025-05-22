#!/bin/bash

# Runs pureNN training on synthetic 4-parameter dataset. Usage on Slurm:
# sbatch slurm_scripts/3a_synthetic_purenn.sh
# (To change the number of GPUs, change --gpus and --num_CPU arguments to that number.)
# (To run on CPU, remove the --gpus line and set --num_CPU to the number of CPUs.)
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the full partition (CPU only)
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J 3a_synthetic_purenn
# Request 4 GPUs 
# #SBATCH --gpus 4
# Request 4 CPU cores (8 hyperthreads).
#SBATCH -c 8
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 80GB RAM
#SBATCH --mem=80GB
# Request a walltime limit of 72 hours
#SBATCH -t 72:00:00

# Activate environment (virtualenv version)
source .venv/bin/activate


for LR in 1e-3
do
    for WD in 0
    do
        for FOLD in 1 2 3 4 5
        do
            if [ $FOLD -eq 1 -a $LR = 1e-2 ]; then
                PLOT_STR="--plot"
            else
                PLOT_STR=""
            fi
            SEED=$FOLD

            python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --optimizer AdamW --lr $LR --weight_decay $WD \
                --seed $SEED --init default \
                --model nn_only --num_layers 3 --residual --activation leaky_relu --use_bn \
                --features ten --para_to_predict four --labels synthetic_function --label_noise_std 0 --standardize_output \
                --losses smooth_l1 --lambdas 1 \
                --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 11.5 --note "3A_SYNTHETIC_PURENN" $PLOT_STR
        done
    done
done

# # Start the Python Code
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
#                 --activation leaky_relu --use_bn --categorical embedding --embed_dim 5 --pos_enc none \
#                 --losses smooth_l1 param_reg --loss_weighting manual --lambdas 1 100 \
#                 --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 11.5 --note "BINN_EXAMPLE"
#         done
#     done
# done