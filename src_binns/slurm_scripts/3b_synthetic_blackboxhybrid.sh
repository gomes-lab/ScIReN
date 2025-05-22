#!/bin/bash

# Runs Blackbox-Hybrid training on synthetic 4-parameter dataset. Usage on Slurm:
# sbatch slurm_scripts/3b_synthetic_blackboxhybrid.sh
# (To change the number of GPUs, change --gpus and --num_CPU arguments to that number.)
# (To run on CPU, remove the --gpus line and set --num_CPU to the number of CPUs.)
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the regular partition (CPU only)
#SBATCH -p regular
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J 3b_synthetic_blackboxhybrid
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
cd /mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/src_binns
source .venv/bin/activate


for LR in 1e-4
do
    for WD in 1e-4
    do
        for TEMP in 1
        do
            for PREG in 0
            do
                for FOLD in 1 2 3 4 5
                do
                    if [ $FOLD -eq 1 -a $LR = 1e-4 ]; then
                        PLOT_STR="--plot"
                    else
                        PLOT_STR=""
                    fi
                    SEED=$FOLD

                    python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                        --optimizer AdamW --lr $LR --weight_decay $WD \
                        --seed $SEED --init default --min_temp $TEMP --max_temp $TEMP \
                        --model new_mlp --num_layers 3 --residual --activation leaky_relu --use_bn \
                        --features ten --para_to_predict four --labels synthetic_function --label_noise_std 0 \
                        --losses smooth_l1 param_reg --lambdas 1 $PREG --param_constraint sigmoid \
                        --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 11.5 --note "3B_SYNTHETIC_BLACKBOXHYBRID"
                done
            done
        done
    done
done
