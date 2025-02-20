export CUDA_VISIBLE_DEVICES=0,1
lr=5e-3
seed=39  
output_dir="variant_causal_eqtl/output_omnireg_10k"
mkdir -p "$output_dir"
torchrun --nproc_per_node=2 variant_train.py  \
        --model_weights_path '../model_and_data/pytorch_model.bin'\
        --kmer -1 \
        --run_name "Hierarchical_${lr}_rna_data0_seed${seed}" \
        --model_max_length 2000 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate "${lr}" \
        --num_train_epochs 10 \
        --seed "${seed}" \
        --fp16 \
        --weight_decay 0.01 \
        --save_steps 2500 \
        --output_dir  "$output_dir" \
        --lr_scheduler_type cosine \
        --warmup_steps 1200 \
        --logging_steps 50 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --eval_steps 250 \
        --evaluation_strategy steps \

    


