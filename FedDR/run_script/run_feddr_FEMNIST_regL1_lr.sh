#!/usr/bin/env bash

EXP_ID='test_feddr_FEMNIST_regl1_lr'
DATASET='FEMNIST'
NUM_ROUND=200
BATCH_SIZE=10
NUM_EPOCH=20
CLIENT_PER_ROUND=50
MODEL='ann'

cd ../

# lr=0.001
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.001 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.001'

# lr=0.003
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.003 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.003'

# lr=0.005
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.005 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.005'

# lr=0.008
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.008 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.008'

# lr=0.01
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=0.01 --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=$NUM_EPOCH --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='lr0.01'