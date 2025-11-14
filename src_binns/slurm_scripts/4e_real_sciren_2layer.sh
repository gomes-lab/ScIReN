#!/bin/bash

# Runs BINN training on GPUs, as a Slurm job. Usage:
# sbatch slurm_scripts/4e_real_sciren_2layer.sh
# (To change the number of GPUs, change --gpus and --num_CPU arguments to that number.)
# (To run on CPU, remove the --gpus line and set --num_CPU to the number of CPUs.)
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the regular partition (CPU only)
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J 4e_real_sciren_2layer
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


for LR in 1e-2
do
    for LAM1 in 1
    do
        for LAM2 in 1
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

                    python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                        --optimizer AdamW --lr $LR --weight_decay 0 --batch_size 32 \
                        --seed $SEED --init default --min_temp 1 --max_temp 1 \
                        --features ten --labels real \
                        --model kan --num_layers 1 \
                        --kan_grid 30 --kan_update_grid 1 --kan_grid_margin 2.0 --kan_base_fun identity --kan_affine_trainable --kan_absolute_deviation \
                        --losses smooth_l1 param_reg param_violation kan_l1 kan_entropy kan_coefdiff kan_coefdiff2 --lambdas 1 0 1000 $LAM1 $LAM2 0 $LAM3 \
                        --param_constraint hardsigmoid \
                        --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note 4D_REAL_SCIREN_2LAYER $PLOT_STR

                done
            done
        done
    done
done
