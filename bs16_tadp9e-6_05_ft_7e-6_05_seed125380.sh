python3 multiple_choice/test_mc_new50257.py \
  --task_name casehold \
  --model_name_or_path gpt2-xl \
  --data_dir data/casehold \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 128 \
  --learning_rate 7e-6 \
  --num_train_epochs 3.0 \
  --output_dir logs/casehold/bs16_tadp9e-6_05_ft_7e-6_05_seed125380 \
  --overwrite_output_dir \
  --overwrite_cache False \
  --logging_steps 1 \
  --gradient_accumulation_steps 16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --evaluation_strategy steps \
  --weight_decay 0.5 \
  --seed 125380 \
  --eval_steps 100 \
  --save_steps 2500 \
  --weight "pytorch_model_50257_decay0.5_lr9e-6.bin"