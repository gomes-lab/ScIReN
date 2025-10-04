#!/bin/bash

# Runs Blackbox-Hybrid (hardsigmoid) training on real labels. Usage on Slurm:
# sbatch slurm_scripts/4c_real_blackboxhybrid_hardsigmoid.sh
# (To change the number of GPUs, change --gpus and --num_CPU arguments to that number.)
# (To run on CPU, remove the --gpus line and set --num_CPU to the number of CPUs.)
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the regular partition (CPU only)
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J 4c_real_blackboxhybrid_hardsigmoid
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

# # Activate environment (virtualenv version)
source .venv/bin/activate


for LR in 1e-4 1e-3 1e-2 1e-1
do
    for WD in 0 1e-4
    do
        for TEMP in 1
        do
            for PREG in 0
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
                        --seed $SEED --init default --min_temp $TEMP --max_temp $TEMP \
                        --model new_mlp --num_layers 1 \
                        --features ten --labels real \
                        --losses smooth_l1 param_reg param_violation --lambdas 1 $PREG 1000 --param_constraint hardsigmoid \
                        --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 71.5 --note "4G_LINEAR" $PLOT_STR
                done
            done
        done
    done
done

# for LR in 1e-4 1e-3 1e-2 1e-1
# do
#     for WD in 0
#     do
#         for TEMP in 1
#         do
#             for PREG in 0
#             do
#                 for FOLD in 1
#                 do
#                     if [ $FOLD -eq 1 -a $LR = 1e-2 ]; then
#                         PLOT_STR="--plot"
#                     else
#                         PLOT_STR=""
#                     fi
#                     SEED=$FOLD

#                     python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
#                         --optimizer AdamW --lr $LR --weight_decay $WD \
#                         --seed $SEED --init default --min_temp $TEMP --max_temp $TEMP \
#                         --model new_mlp --num_layers 3 --residual --activation leaky_relu --use_bn \
#                         --features ten --labels real \
#                         --losses smooth_l1 param_reg param_violation --lambdas 1 $PREG 1000 --param_constraint hardsigmoid \
#                         --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 71.5 --note "REAL_BINN_HARDSIGMOID" $PLOT_STR
#                 done
#             done
#         done
#     done
# done
