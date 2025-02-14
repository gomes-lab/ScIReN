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
