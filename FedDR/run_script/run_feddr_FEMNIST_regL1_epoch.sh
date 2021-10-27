#!/usr/bin/env bash

EXP_ID='test_feddr_FEMNIST_regl1_epoch'
DATASET='FEMNIST'
LEARNING_RATE=0.003
NUM_ROUND=200
BATCH_SIZE=10
CLIENT_PER_ROUND=50
MODEL='ann'

cd ../

# nepoch=5
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=5 --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='epoch05'

# nepoch=10
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=10 --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='epoch10'

# nepoch=15
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=15 --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='epoch15'

# nepoch=20
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=20 --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='epoch20'

# nepoch=30
python3  -u main.py --dataset=$DATASET --optimizer='feddr' --exp_id=$EXP_ID  \
            --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
            --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 --batch_size=$BATCH_SIZE \
            --num_epochs=30 --model=$MODEL \
            --eta=1000 \
            --alpha=1.95 \
            --reg_type='l1_norm' --reg_coeff=0.01 --log_suffix='epoch30'