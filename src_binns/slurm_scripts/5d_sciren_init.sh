#!/bin/bash

# Test sciren init

# Request the regular partition (CPU only)
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J 5d_sciren_init
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

# 1 CPU
for LR in 1e-2
do
    for LAM0 in 1 10 100 1000
    do
        for LAM1 in 1
        do
            LAM2=$LAM1
            for LAM3 in 1000
            do
                for TEMP in 1
                do
                    for FOLD in 1
                    do
                        for NOISE in 0.3 
                        do
                            for SEED in 1
                            do
                                if [ $FOLD -eq 1 ]; then
                                    PLOT_STR="--plot"
                                else
                                    PLOT_STR=""
                                fi

                                python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                                    --optimizer AdamW --lr $LR --weight_decay 0 --batch_size 32 --n_epochs 100 \
                                    --seed $SEED --init default --min_temp $TEMP --max_temp $TEMP  \
                                    --features ten --labels real \
                                    --model kan --num_layers 1 \
                                    --kan_grid 30 --kan_update_grid 1 --kan_grid_margin 2.0 --kan_base_fun identity --kan_noise $NOISE --use_bn \
                                    --losses smooth_l1 unconstrained_param kan_l1 kan_entropy kan_coefdiff kan_coefdiff2 kan_diversity --lambdas 1 $LAM0 $LAM1 $LAM2 0 $LAM3 0 \
                                    --param_constraint hardsigmoid \
                                    --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "5D_SCIREN_UNCONLOSS" $PLOT_STR
                            done
                        done
                    done
                done
            done
        done
    done
done  #--kan_affine_trainable


# for LR in 1e-2
# do
#     for LAM1 in 0.1
#     do
#         for LAM2 in 1
#         do
#             for LAM3 in 1000
#             do
#                 for TEMP in 0.2
#                 do
#                     for FOLD in 1
#                     do
#                         for SEED in 1 2 3
#                         do
#                             if [ $FOLD -eq 1 ]; then
#                                 PLOT_STR="--plot"
#                             else
#                                 PLOT_STR=""
#                             fi

#                             python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
#                                 --optimizer AdamW --lr $LR --weight_decay 0 --batch_size 32 --n_epochs 200 \
#                                 --seed $SEED --init default --min_temp $TEMP --max_temp $TEMP --final_bias uniform2_init \
#                                 --features ten --labels real \
#                                 --model kan --num_layers 1 \
#                                 --kan_grid 30 --kan_update_grid 1 --kan_grid_margin 2.0 --kan_base_fun identity \
#                                 --losses smooth_l1 param_reg param_violation kan_l1 kan_entropy kan_coefdiff kan_coefdiff2 kan_diversity --lambdas 1 100 1000 $LAM1 $LAM2 0 $LAM3 1 \
#                                 --param_constraint hardsigmoid \
#                                 --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "5D_SCIREN_INIT_TEMP=${TEMP}_DIVERSITYLOSS1" $PLOT_STR
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done  #--kan_affine_trainable

