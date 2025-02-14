# Runs BINN training on GPUs, interactively.
# 1) Request job. On slurm, an example command to request 4 GPUs is:
# srun -p full -n 1 -c 8 --time=72:00:00 --mem-per-cpu=10G --gpus 4 --pty /bin/bash -l
# 2) Run the script. (--num_CPU should be set to number of GPUs if available, otherwise the number of CPUs.)
# ./run_interactive.sh
# TODO Figure out what is wrong with the vertical mixing


# TUNING: Default init
for LR in 1e-4 1e-3 1e-2 1e-1
do
    for FOLD in 1 2 3 4 5
    do
        for SEED in 0 1 2
        do
            python3 binns_DDP.py --data_seed 12345 --n_datapoints 400 --split grid2 --cross_val_idx $FOLD --n_folds 5 \
                --lr $LR --optimizer AdamW --weight_decay 0 \
                --seed $SEED --init default --min_temp 1 --max_temp 1 --standardize_input \
                --n_epochs 200 --patience 100 --model new_mlp --vertical_mixing original --vectorized "true" \
                --activation leaky_relu --use_bn --categorical one_hot --pos_enc none \
                --losses l2 --loss_weighting manual --lambdas 1 \
                --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "BINN_GRID2_L2ONLY"
        done
    done
done


# # TUNING: Default init
# for LR in 1e-4 1e-3 1e-2 1e-1 1
# do
#     for FOLD in 1
#     do
#         for SEED in 0 1 2
#         do
#             python3 binns_DDP.py --data_seed 12345 --n_datapoints 400 --split vertical --cross_val_idx $FOLD --n_folds 5 \
#                 --lr $LR --optimizer AdamW --weight_decay 0 \
#                 --seed $SEED --init default --min_temp 1 --max_temp 1 --standardize_input \
#                 --n_epochs 200 --patience 100 --model new_mlp --vertical_mixing original --vectorized "true" \
#                 --leaky_relu --use_bn --embed_dim 5 --pos_enc none \
#                 --losses l1 param_reg --loss_weighting manual --lambdas 1 100 \
#                 --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "BINN_VERTICAL400_TUNING"
#         done
#     done
# done



# # TUNING: Original training
# for LR in 1e-4 1e-3 1e-2 1e-1 1
# do
#     for FOLD in 1
#     do
#         for SEED in 0 1 2
#         do
#             python3 binns_DDP.py --data_seed 12345 --n_datapoints 200 --split horizontal --cross_val_idx $FOLD --n_folds 5 \
#                 --lr $LR --optimizer AdamW --weight_decay 0 \
#                 --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
#                 --n_epochs 200 --patience 100 --model new_mlp --vertical_mixing original \
#                 --leaky_relu --use_bn --embed_dim 5 --pos_enc none \
#                 --losses l1 param_reg --loss_weighting manual --lambdas 1 100 \
#                 --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "BINN_ORIGINAL_TUNING"
#         done
#     done
# done




# --leaky_relu --use_bn

# # +Drop
# for LR in 1e-2 3e-2 3e-3
# do
#     for SEED in 0 1 2
#     do
#         python3 binns_DDP.py --data_seed 12345 --n_datapoints 200 --split west \
#             --lr $LR --optimizer AdamW --weight_decay - --dropout_prob 0.1 --seed $SEED  \
#             --n_epochs 200 --patience 100 --model new_mlp --vertical_mixing original --leaky_relu --use_bn \
#             --embed_dim 5 --pos_enc early \
#             --losses l1 param_reg --loss_weighting manual --lambdas 1 100 \
#             --num_CPU 5 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note WEST_BINN_STD_DEBUG_WD
#     done
# done

# for LR in 1e-3 1e-2
# do
#     for SEED in 0 1 2
#     do
#         python3 binns_DDP.py --lr $LR --weight_decay 0 --seed $SEED --n_datapoints 1000 \
#             --n_epochs 200 --patience 20 --model new_mlp --vertical_mixing original \
#             --use_bn --embed_dim 10 --pos_enc early \
#             --losses l1 param_reg --loss_weighting manual --lambdas 1 10 \
#             --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note STD1000
#     done
# done



