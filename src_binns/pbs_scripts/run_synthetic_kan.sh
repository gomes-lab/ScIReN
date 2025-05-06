#!/bin/bash
#PBS -A UCOR0092
#PBS -N BINN_syntheticfunction_KAN
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=128

# Usage: qsub pbs_scripts/run_synthetic_kan.sh
# Use scratch for temporary files to avoid space limits in /tmp
# export TMPDIR="/glade/scratch/$USER/temp"
# mkdir -p $TMPDIR
# Load modules to match compile-time environment
# module purge
# module load conda
# module load cuda

# # Activate environment (virtualenv version)
cd /glade/work/joshuaf/BINNS/src_binns
source .venv/bin/activate

# (Conda version)
# conda activate BINN_310_CPU'

for LR in 1e-2
do
    for LAM1 in 1
    do
        for LAM3 in 1000
        do
            for FOLD in 1
            do
                if [ $FOLD -eq 1 -a $LR = 1e-2 ]; then
                    PLOT_STR="--plot"
                else
                    PLOT_STR=""
                fi
                SEED=$FOLD
                LAM2=$(echo "$LAM1*2" | bc)
                echo $LAM2

                python3 binns_DDP.py --data_seed 12345 --split grid2  --cross_val_idx $FOLD --representative_sample --n_folds 5 \
                    --lr $LR --optimizer AdamW --batch_size 32 \
                    --seed $SEED --init default --min_temp 1 --max_temp 1 \
                    --n_epochs 200 --patience 100 --model kan \
                    --num_layers 1 --features ten --para_to_predict four --labels synthetic_function  --label_noise_std 0 \
                    --losses smooth_l1 param_reg param_violation kan_l1 kan_entropy kan_coefdiff kan_coefdiff2 --lambdas 1 1 1000 $LAM1 $LAM2 0 $LAM3 \
                    --param_constraint hardsigmoid \
                    --kan_grid 30 --kan_update_grid 1 --kan_grid_margin 2.0 --kan_base_fun identity --kan_affine_trainable \
                    --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "KAN_SYNTHETICFUNCTION_GRID30" $PLOT_STR
                exit
            done
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
