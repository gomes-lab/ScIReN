for LR in 1e-4
do
    for SEED in 0
    do
        python3 binns_singleprocess.py --lr $LR --weight_decay 0 --seed $SEED --n_epochs 500 --patience 10
    done
done