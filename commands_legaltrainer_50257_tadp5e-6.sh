python3 multiple_choice/run_mc_50257legaltrainer.py \
  --task_name casehold \
  --model_name_or_path gpt2-xl \
  --data_dir data/casehold \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 128 \
  --learning_rate 5e-6 \
  --num_train_epochs 6.0 \
  --output_dir logs/casehold/50257tadp_5e-6 \
  --overwrite_output_dir \
  --overwrite_cache False \
  --logging_steps 1 \
  --gradient_accumulation_steps 64 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --evaluation_strategy steps \
  --weight_decay 0.01 \
  --seed 42 \
  --eval_steps 200 \
  --save_steps 1500 \
  --weight "pytorch_model_lr1e-5_wd1.0_ckpt1680.bin"
