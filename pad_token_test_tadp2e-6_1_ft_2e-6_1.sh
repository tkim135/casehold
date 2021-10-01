python3 multiple_choice/pad_tokentest.py \
  --task_name casehold \
  --model_name_or_path gpt2-xl \
  --data_dir data/casehold \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 128 \
  --learning_rate 5e-6 \
  --num_train_epochs 1.5 \
  --output_dir logs/casehold/pad_token_test_tadp2e-6_1_ft_5e-6_1 \
  --overwrite_output_dir \
  --overwrite_cache False \
  --logging_steps 1 \
  --gradient_accumulation_steps 16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --evaluation_strategy steps \
  --weight_decay 1.0 \
  --seed 42 \
  --eval_steps 100 \
  --save_steps 3972 \
  --weight "snckpt_model_decay1.0_lr2e-6.bin"
