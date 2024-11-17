# Runs BINN training on GPUs, interactively.
# 1) Request job. On slurm, an example command to request 4 GPUs is:
# srun -p full -n 1 -c 8 --time=120:00:00 --mem-per-cpu=10G --gpus 4 --pty /bin/bash -l
# 2) Run the script. (--num_CPU should be set to number of GPUs if available, otherwise the number of CPUs.)
# ./run_interactive.sh
# TODO Figure out what is wrong with the vertical mixing
for LR in 1e-2
do
    for SEED in 0
    do
        python3 binns_DDP.py --lr $LR --weight_decay 1e-3 --seed $SEED --n_datapoints 500 \
            --n_epochs 20 --patience 20 --model new_mlp --vertical_mixing original --use_bn --embed_dim 5 --pos_enc early \
            --losses l1 param_reg --loss_weighting manual --lambdas 1 100 \
            --num_CPU 1 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note DEBUG_VMIXING
    done
done



