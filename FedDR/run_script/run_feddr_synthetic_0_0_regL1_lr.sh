#!/usr/bin/env bash

EXP_ID='test_feddr_synthetic_0_0_regl1_lr'
DATASET='synthetic_0_0'
NUM_ROUND=10
BATCH_SIZE=10
NUM_EPOCH=20
CLIENT_PER_ROUND=10
MODEL='ann'

cd ../

# lr=0.0025
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.0025 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=100 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.0025'

# lr=0.005
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.005 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=100 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.005'

# lr=0.0075
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.0075 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=100 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.0075'

# lr=0.01
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.01 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=100 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.01'

# lr=0.025
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.025 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=100 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.025'