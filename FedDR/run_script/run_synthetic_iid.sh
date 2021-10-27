#!/usr/bin/env bash

EXP_ID='test_synthetic_iid'
DATASET='synthetic_iid'
LEARNING_RATE=0.01
NUM_ROUND=200
BATCH_SIZE=10
NUM_EPOCH=20
CLIENT_PER_ROUND=10
MODEL='ann'

cd ../

# fedavg
python3  -u main.py --dataset=$DATASET --optimizer='fedavg' --exp_id=$EXP_ID  \
            --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 \
            --batch_size=$BATCH_SIZE --num_epochs=$NUM_EPOCH --model=$MODEL

# fedprox
python3  -u main.py --dataset=$DATASET --optimizer='fedprox' --exp_id=$EXP_ID  \
            --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 \
            --batch_size=$BATCH_SIZE --num_epochs=$NUM_EPOCH --model=$MODEL \
            --mu=0.1

# fedpd
python3  -u main.py --dataset=$DATASET --optimizer='fedpd' --exp_id=$EXP_ID  \
            --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
            --clients_per_round=0 --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=10

# feddr
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=100 \
            --alpha=1.95 \
            --reg_type='none'