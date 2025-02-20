#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
output_dir='pretrain_output'

mkdir -p "$output_dir"
per_device_train_batch_size=4
gradient_accumulation_steps=4
weight_decay=1e-4
model_max_length=4096
seed=42
learning_rate=1e-4
warmup_steps=6000

echo "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 pretrain.py" > "$output_dir/hyperparameters.txt"
echo "--input_seq_len $input_seq_len" >> "$output_dir/hyperparameters.txt"
echo "--per_device_train_batch_size $per_device_train_batch_size" >> "$output_dir/hyperparameters.txt"
echo "--gradient_accumulation_steps $gradient_accumulation_steps" >> "$output_dir/hyperparameters.txt"
echo "--weight_decay $weight_decay" >> "$output_dir/hyperparameters.txt"
echo "--model_max_length $model_max_length" >> "$output_dir/hyperparameters.txt"
echo "--seed $seed" >> "$output_dir/hyperparameters.txt"
echo "--learning_rate $learning_rate" >> "$output_dir/hyperparameters.txt"
cp "pretrain_local.py" "$output_dir"
cp "rewrite_hierarchical_transformer.py" "$output_dir"

 
torchrun --nnodes 1 --nproc_per_node=4  pretrain.py \
    --output_dir "$output_dir" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay \
    --model_max_length $model_max_length \
    --seed $seed \
    --learning_rate $learning_rate \
    --lr_scheduler_type cosine \
    --warmup_steps  $warmup_steps \
    --num_train_epochs 4 \
    --logging_steps 10 \
    --save_steps 5000 \
    --do_train \
    --fp16 \
    
