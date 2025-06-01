#!/bin/bash

# Runs Blackbox-Hybrid (hardsigmoid) training on synthetic 4-parameter dataset. Usage on Slurm:
# sbatch slurm_scripts/3d_synthetic_sciren_1layer.sh
# (To change the number of GPUs, change --gpus and --num_CPU arguments to that number.)
# (To run on CPU, remove the --gpus line and set --num_CPU to the number of CPUs.)
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the full partition (CPU only)
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J 3d_synthetic_sciren_1layer
# Request 4 GPUs 
# #SBATCH --gpus 4
# Request 4 CPU cores (8 hyperthreads).
#SBATCH -c 1
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 80GB RAM
#SBATCH --mem=80GB
# Request a walltime limit of 72 hours
#SBATCH -t 72:00:00

# Activate environment (virtualenv version)
source .venv/bin/activate


for LR in 1e-2
do
    for LAM1 in 10
    do
        for LAM3 in 100
        do
            for FOLD in 1 2 3 4 5
            do
                if [ $FOLD -eq 1 ]; then
                    PLOT_STR="--plot"
                else
                    PLOT_STR=""
                fi
                SEED=$FOLD
                LAM2=$LAM1

                # # LAM2=$(echo "$LAM1*2" | bc)
                # # https://stackoverflow.com/questions/19075671/how-do-i-use-shell-variables-in-an-awk-script
                # LAM2=`awk -v var="$LAM1" 'BEGIN{ print 2.0 * var }'`  # $(( 2*LAM1 ))
                # echo $LAM2
                python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --optimizer AdamW --lr $LR --weight_decay 0 \
                --seed $SEED --init default --min_temp 1 --max_temp 1 \
                --features ten --para_to_predict four --labels synthetic_function --label_noise_std 0 \
                --model kan --num_layers 1 --kan_grid 30 --kan_update_grid 1 --kan_grid_margin 2.0 --kan_base_fun identity --kan_affine_trainable --kan_absolute_deviation \
                --losses smooth_l1 param_reg param_violation kan_l1 kan_entropy kan_coefdiff kan_coefdiff2 --lambdas 1 0 1000 $LAM1 $LAM2 0 $LAM3 \
                --param_constraint hardsigmoid \
                --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 11.5 --note "3D_SYNTHETIC_SCIREN_1LAYER" $PLOT_STR
            done
        done
    done
done


# (Conda version)
# conda activate BINN_310_CPU'

# for LR in 1e-2
# do
#     for LAM1 in 1 10 100
#     do
#         for LAM3 in 100 1000 10000
#         do
#             for FOLD in 1
#             do
#                 if [ $FOLD -eq 1 -a $LR = 1e-2 ]; then
#                     PLOT_STR="--plot"
#                 else
#                     PLOT_STR=""
#                 fi
#                 SEED=$FOLD
#                 # LAM2=$(echo "$LAM1*2" | bc)
#                 # echo $LAM2

#                 python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2  --cross_val_idx $FOLD --n_folds 5 \
#                     --optimizer AdamW --lr $LR --weight_decay 0 \
#                     --seed $SEED --init default --min_temp 1 --max_temp 1 \
#                     --model kan --num_layers 1 \
#                     --features ten --para_to_predict fifteen --labels synthetic_function --label_noise_std 0 \
#                     --losses smooth_l1 param_reg param_violation kan_l1 kan_entropy kan_coefdiff kan_coefdiff2 --lambdas 1 0 1000 $LAM1 $LAM1 $LAM3 0 \
#                     --param_constraint hardsigmoid \
#                     --kan_grid 30 --kan_update_grid 1 --kan_grid_margin 2.0 --kan_base_fun zero --kan_affine_trainable \
#                     --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "KAN_SYNTHETICFUNCTION_FIFTEEN_ZERO" $PLOT_STR
#             done
#         done
#     done
# done

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
