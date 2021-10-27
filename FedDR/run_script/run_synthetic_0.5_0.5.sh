#!/usr/bin/env bash

EXP_ID='test_synthetic_0.5_0.5'
DATASET='synthetic_0.5_0.5'
LEARNING_RATE=0.01
NUM_ROUND=200
BATCH_SIZE=10
NUM_EPOCH=20
CLIENT_PER_ROUND=10
MODEL='ann'

cd ../

# fedavg
python3  -u main.py --dataset=$DATASET --optimizer='fedavg' --exp_id=$EXP_ID  \
            --learning_rate=0.01 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 \
            --batch_size=$BATCH_SIZE --num_epochs=$NUM_EPOCH --model=$MODEL

# fedprox
python3  -u main.py --dataset=$DATASET --optimizer='fedprox' --exp_id=$EXP_ID  \
            --learning_rate=0.01 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 \
            --batch_size=$BATCH_SIZE --num_epochs=$NUM_EPOCH --model=$MODEL \
            --mu=1.0

# fedpd
python3  -u main.py --dataset=$DATASET --optimizer='fedpd' --exp_id=$EXP_ID  \
            --learning_rate=0.01 --num_rounds=$NUM_ROUND \
            --clients_per_round=0 --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=1

# feddr
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.01 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=500 \
            --alpha=1.95 \
            --reg_type='none'