
# srun -p aida -n 1 -c 32 --time=120:00:00 --mem-per-cpu=20G --gpus a100:4 --pty /bin/bash -l

for LR in 1e-3
do
    for SEED in 0
    do
        python3 binns_DDP.py --lr $LR --weight_decay 0 --seed $SEED --n_epochs 500 --patience 20 --model lipmlp --lambda_lipschitz $1 --note EMBED5
    done
done

# ./run_lipmlp.sh 100  [1] embed 15  [2] embed 5