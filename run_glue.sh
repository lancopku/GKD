# an example of shell cmd for run_glue.py

export TASK_NAME=qqp # or sst2, mnli, qnli

python run_glue.py \
--model_name_or_path bert-base-uncased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 4 \
--output_dir ./teacher/$TASK_NAME/ \
--overwrite_output_dir \
--evaluation_strategy steps \
--load_best_model_at_end \
--save_strategy steps \
--metric_for_best_model accuracy \
--save_total_limit 1 \
--seed 42
