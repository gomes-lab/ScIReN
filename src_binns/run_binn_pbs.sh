#!/bin/bash
#PBS -A UOKL0017
#PBS -N DDP_BINN_Resume
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8

# Usage: qsub run_binn_pbs.sh
# Use scratch for temporary files to avoid space limits in /tmp
export TMPDIR="/glade/scratch/$USER/temp"
mkdir -p $TMPDIR
# Load modules to match compile-time environment
module purge
module load conda
module load cuda

# Activate environment (virtualenv version)
cd /glade/work/joshuaf/BINNS/src_binns
source .venv/bin/activate

# (Conda version)
# conda activate BINN_310_CPU

# Start the Python Code
for LR in 1e-3
do
    for FOLD in 1
    do
        for SEED in 0
        do
            python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --lr $LR --optimizer AdamW --weight_decay 0 \
                --seed $SEED --init default --min_temp 1 --max_temp 1 \
                --n_epochs 200 --patience 100 --model new_mlp --vertical_mixing original --vectorized yes \
                --activation leaky_relu --use_bn --categorical embedding --embed_dim 5 --pos_enc none \
                --losses smooth_l1 param_reg --loss_weighting manual --lambdas 1 100 \
                --num_CPU 8 --use_ddp 1 --job_scheduler slurm --time_limit 11.5 --note "BINN_EXAMPLE"
        done
    done
done