python3 multiple_choice/pile_finetune.py \
  --task_name casehold \
  --model_name_or_path gpt2-xl \
  --data_dir data/casehold \
  --cache_dir /import/snvm-sc-scratch2/tonyk/.cache \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 128 \
  --learning_rate 4e-6 \
  --num_train_epochs 1.0 \
  --output_dir /import/ml-sc-scratch2/tonyk/legalgpt/a100/logs/casehold/pile_4e-6_05 \
  --overwrite_output_dir \
  --overwrite_cache False \
  --logging_steps 1 \
  --gradient_accumulation_steps 16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --evaluation_strategy steps \
  --weight_decay 0.5 \
  --seed 42 \
  --eval_steps 100 \
  --save_steps 2400 \
  --weight "/import/snvm-sc-scratch1/urmisht/PD/pile_chkpt.pt"
