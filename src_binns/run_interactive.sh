# Runs BINN training on GPUs, interactively.
# 1) Request job. On slurm, an example command to request 4 GPUs is:
# srun -p aida -n 1 -c 8 --time=120:00:00 --mem-per-cpu=10G --gpus 4 --pty /bin/bash -l
# 2) Run the script. (--num_CPU should be set to number of GPUs if available, otherwise the number of CPUs.)
# ./run_interactive.sh
for LR in 1e-2
do
    for SEED in 0
    do
        python3 binns_DDP.py --lr $LR --weight_decay 1e-3 --seed $SEED --n_epochs 200 --patience 20 \
            --model old_mlp --use_bn --embed_dim 10 --num_CPU 4 --job_scheduler slurm --time_limit 119.5
    done
done