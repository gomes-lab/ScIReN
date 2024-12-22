# Runs pure-NN training on GPUs, interactively.
# 1) Request job. On slurm, an example command to request 4 GPUs is:
# srun -p full -n 1 -c 8 --time=120:00:00 --mem-per-cpu=10G --gpus 4 --pty /bin/bash -l
# 2) Run the script. (--num_CPU should be set to number of GPUs if available, otherwise the number of CPUs.)
# ./run_interactive.sh
# TODO Figure out what is wrong with the vertical mixing. Right now setting simple_two_intercepts only for the "bulk simulations" which are actually meaningless.
for LR in 1e-4
do
    for SEED in 0
    do
        python3 binns_DDP.py --n_datapoints 2000 --data_seed 12345 --split west \
            --lr $LR --optimizer SGD --weight_decay 1e-3 --seed $SEED \
            --n_epochs 200 --patience 50 --model nn_only --vertical_mixing simple_two_intercepts \
            --use_bn --embed_dim 10 --pos_enc early \
            --losses l2 --loss_weighting manual --lambdas 1 \
            --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note EAST2000_NNONLY_L2
    done
done