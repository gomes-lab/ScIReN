# Runs pure-NN training on GPUs, interactively.
# 1) Request job. On slurm, an example command to request 4 GPUs is:
# srun -p full -n 1 -c 8 --time=120:00:00 --mem-per-cpu=10G --gpus 4 --pty /bin/bash -l
# 2) Run the script. (--num_CPU should be set to number of GPUs if available, otherwise the number of CPUs.)
# ./run_interactive.sh
# TODO Figure out what is wrong with the vertical mixing. Right now setting simple_two_intercepts only for the "bulk simulations" which are actually meaningless.

# TUNING: NO STANDARDIZE OUTPUT
for LR in 1e-4 1e-3 1e-2 1e-1 1 10 100
do
    for FOLD in 1
    do
        for SEED in 0 1 2
        do
            python3 binns_DDP.py --data_seed 12345 --n_datapoints 200 --split horizontal --cross_val_idx $FOLD --n_folds 5 \
                --lr $LR --optimizer AdamW --weight_decay 0 \
                --seed $SEED --init default \
                --n_epochs 200 --patience 100 --model nn_only --vertical_mixing simple_two_intercepts \
                --leaky_relu --use_bn --embed_dim 5 --pos_enc none \
                --losses l1 --loss_weighting manual --lambdas 1 \
                --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "NNONLY_TUNING"
        done
    done
done

# TUNING: STANDARDIZE OUTPUT
for LR in 1e-4 1e-3 1e-2 1e-1 1 10 100
do
    for FOLD in 1
    do
        for SEED in 0 1 2
        do
            python3 binns_DDP.py --data_seed 12345 --n_datapoints 200 --split horizontal --cross_val_idx $FOLD --n_folds 5 \
                --lr $LR --optimizer AdamW --weight_decay 0 \
                --seed $SEED --init default --standardize_output \
                --n_epochs 200 --patience 100 --model nn_only --vertical_mixing simple_two_intercepts \
                --leaky_relu --use_bn --embed_dim 5 --pos_enc none \
                --losses l1 --loss_weighting manual --lambdas 1 \
                --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "NNONLY_TUNING_STDOUTPUT"
        done
    done
done

# TUNING: STANDARDIZE OUTPUT+INPUT
for LR in 1e-4 1e-3 1e-2 1e-1 1 10 100
do
    for FOLD in 1
    do
        for SEED in 0 1 2
        do
            python3 binns_DDP.py --data_seed 12345 --n_datapoints 200 --split horizontal --cross_val_idx $FOLD --n_folds 5 \
                --lr $LR --optimizer AdamW --weight_decay 0 \
                --seed $SEED --init default --standardize_output --standardize_input \
                --n_epochs 200 --patience 100 --model nn_only --vertical_mixing simple_two_intercepts \
                --leaky_relu --use_bn --embed_dim 5 --pos_enc none \
                --losses l1 --loss_weighting manual --lambdas 1 \
                --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "NNONLY_TUNING_STDOUTPUTINPUT"
        done
    done
done

# for LR in 1e-2
# do
#     for FOLD in 1 2 3 4 5
#     do
#         for SEED in 0 1 2
#         do
#             python3 binns_DDP.py --data_seed 12345 --n_datapoints 200 --split horizontal --cross_val_idx $FOLD --n_folds 5 \
#                 --lr $LR --optimizer AdamW --weight_decay 0 --seed $SEED \
#                 --n_epochs 200 --patience 100 --model nn_only --vertical_mixing simple_two_intercepts \
#                 --leaky_relu --use_bn --embed_dim 5 --pos_enc none \
#                 --losses l1 --loss_weighting manual --lambdas 1 \
#                 --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "NNONLY"
#             exit
#         done
#     done
# done
