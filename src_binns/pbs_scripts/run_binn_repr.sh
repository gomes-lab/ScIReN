#!/bin/bash
#PBS -A UCOR0092
#PBS -N BINN_std
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=128

# Usage: qsub pbs_scripts/run_binn_repr.sh
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
        for TEMP in 1
        do
            for PREG in 1 100
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
                        --features ten \
                        --model new_mlp --num_layers 3 --residual --activation leaky_relu --use_bn \
                        --losses smooth_l1 param_reg --lambdas 1 $PREG --param_constraint sigmoid \
                        --num_CPU 8 --use_ddp 1 --job_scheduler pbs --time_limit 11.5 --note "BINN_STD_REPR" $PLOT_STR
                done
            done
        done
    done
done
