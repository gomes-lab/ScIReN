#!/bin/bash

# Runs pureNN training on real labels. Usage on Slurm:
# sbatch slurm_scripts/4a_real_purenn.sh
# (To change the number of GPUs, change --gpus and --num_CPU arguments to that number.)
# (To run on CPU, remove the --gpus line and set --num_CPU to the number of CPUs.)
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the regular partition (CPU only)
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J 5a_realfull_purenn
# Request 4 GPUs 
# #SBATCH --gpus 4
# Request 4 CPU cores (8 hyperthreads).
#SBATCH -c 64
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 80GB RAM
#SBATCH --mem=200GB
# Request a walltime limit of 72 hours
#SBATCH -t 72:00:00

# Activate environment (virtualenv version)
source .venv/bin/activate


for LR in 1e-4 1e-3 1e-2
do
    for WD in 0
    do
        for FOLD in 1
        do
            if [ $FOLD -eq 1 ]; then
                PLOT_STR="--plot"
            else
                PLOT_STR=""
            fi
            SEED=$FOLD

            python3 binns_DDP.py --data_seed 12345 --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --optimizer AdamW --lr $LR --weight_decay $WD --batch_size 16 \
                --seed $SEED --init default \
                --model nn_only --num_layers 3 --residual --activation leaky_relu --use_bn \
                --features ten --labels real --standardize_output \
                --losses smooth_l1 --lambdas 1 \
                --num_CPU 32 --use_ddp 1 --job_scheduler slurm --time_limit 11.5 --note "5A_REALFULL_PURENN_BATCHDEBUG" $PLOT_STR
            exit
        done
    done
done
