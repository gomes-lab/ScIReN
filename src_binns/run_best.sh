for LR in 1e-3
do
    for TEMP in 1
    do
        for FOLD in 1
        do
            SEED=$FOLD
            python3 binns_DDP.py --data_seed 12345 --representative_sample --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --lr $LR --optimizer AdamW --weight_decay 0 \
                --seed $SEED --init default --min_temp $TEMP --max_temp $TEMP \
                --n_epochs 200 --patience 100 --model new_mlp  \
                --num_layers 3 --residual --width 256 --bias_only_epochs 30 \
                --activation leaky_relu --use_bn --categorical one_hot --pos_enc none \
                --losses smooth_l1 --lambdas 1 \
                --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "BINN_3layers"
        done
    done
done


# srun -p aida -n 1 -c 32 --time=120:00:00 --mem-per-cpu=20G --gpus a100:4 --pty /bin/bash -l



# # Best-performing run so far
# for LR in 1e-2
# do
#     for LAM in 1
#     do
#         for SEED in 0 1 2
#         do
#             # Note: lr is learning rate. 
#             # "--model new_mlp --lambda_lipschitz $LAM" implies using the new MLP structure with spectral regularization (strength $LAM)
#             # "--categorical one_hot" means using a one-hot encoding for categorical variables
#             # "--use_bn" means to use batchnorm
#             python3 binns_DDP.py --lr $LR --weight_decay 0 --seed $SEED --n_epochs 500 --patience 50 \
#                 --model new_mlp --lambda_lipschitz $LAM --note SPECTRALREG --categorical one_hot --use_bn
#         done
#     done
# done
