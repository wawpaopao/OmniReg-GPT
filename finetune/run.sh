export CUDA_VISIBLE_DEVICES=0

data_path=/path/to/your/dataset.h5
lr=2e-5
seed=42

model_weights_paths=(
    '/path/to/your/model_weights.bin'
)

for model_weights_path in "${model_weights_paths[@]}"; do
    model_weights_name=$(basename "$model_weights_path" | sed 's/\//_/g')
    
    output_dir="/path/to/output_directory/${lr}_seed${seed}"
    mkdir -p "$output_dir"
    
    echo "Training with model_weights_path: $model_weights_path"
    python  finetune.py \
        --model_weights_path "$model_weights_path" \
        --kmer -1 \
        --run_name "Task_${lr}_${data}_seed${seed}_batch_32" \
        --model_max_length 5000 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate "${lr}" \
        --num_train_epochs 5 \
        --seed "${seed}" \
        --fp16 \
        --weight_decay 0.05 \
        --save_steps 2000 \
        --output_dir "$output_dir" \
        --lr_scheduler_type cosine \
        --warmup_steps 1000 \
        --logging_steps 25 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --eval_steps 2000 \
        --evaluation_strategy steps
done
