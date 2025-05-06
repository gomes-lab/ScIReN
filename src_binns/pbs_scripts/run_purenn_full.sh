#!/bin/bash
#PBS -A UCOR0092
#PBS -N BINN_pureNN_FULL
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=128

# Usage: qsub pbs_scripts/run_purenn_repr.sh
# # Use scratch for temporary files to avoid space limits in /tmp
# export TMPDIR="/glade/scratch/$USER/temp"
# mkdir -p $TMPDIR
# # Load modules to match compile-time environment
# module purge
# module load conda
# module load cuda

# Activate environment (virtualenv version)
cd /glade/work/joshuaf/BINNS/src_binns
source .venv/bin/activate

# (Conda version)
# conda activate BINN_310_CPU

for LR in 1e-4 1e-3 1e-2 1e-1
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

            python3 binns_DDP.py --data_seed 12345 --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --optimizer AdamW --lr $LR --weight_decay $WD \
                --seed $SEED --init default \
                --features ten --standardize_output \
                --model nn_only --num_layers 3 --residual --activation leaky_relu --use_bn \
                --losses smooth_l1 --lambdas 1 \
                --num_CPU 128 --use_ddp 1 --job_scheduler pbs --time_limit 11.5 --note "PURENN_FULL" $PLOT_STR
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