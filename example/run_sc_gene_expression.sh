export CUDA_VISIBLE_DEVICES=2
lr=5e-3
seed=39

model_weights_paths=(
    '../model_and_data/pytorch_model.bin'
)

for model_weights_path in "${model_weights_paths[@]}"; do

    model_weights_name=$(basename "$model_weights_path" | sed 's/\//_/g')
    
    output_dir="sc_gene_expression"
    mkdir -p "$output_dir"
    
    echo "Training with model_weights_path: $model_weights_path"
    python  sc_gene_expression_train.py \
        --model_weights_path "$model_weights_path" \
        --kmer -1 \
        --run_name "Hierarchical_${lr}_nvwa_${data}_seed${seed}" \
        --model_max_length 3200 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 4 \
        --learning_rate "${lr}" \
        --num_train_epochs 30 \
        --seed "${seed}" \
        --fp16 \
        --weight_decay 0.05 \
        --save_steps 2000 \
        --output_dir "$output_dir" \
        --lr_scheduler_type cosine \
        --warmup_steps 500 \
        --logging_steps 25 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --eval_steps 1000 \
        --evaluation_strategy steps
done
