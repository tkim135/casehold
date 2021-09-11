python3 multiple_choice/run_mc_legaltrainer.py \
  --task_name casehold \
  --model_name_or_path logs/casehold/gpt_1e-5_001/checkpoint-1500 \
  --data_dir data/casehold \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 128 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --output_dir logs/casehold/gpt_1e-5_001 \
  --overwrite_output_dir \
  --overwrite_cache False \
  --logging_steps 1 \
  --gradient_accumulation_steps 64 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --evaluation_strategy steps \
  --weight_decay 0.01 \
  --seed 42 \
  --eval_steps 200