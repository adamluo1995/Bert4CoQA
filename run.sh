CUDA_VISIBLE_DEVICES=0 \
python run_coqa.py \
--type bert \
--bert_model ../bert-base-uncased/ \
--do_train \
--do_predict \
--output_dir tmp \
--train_file coqa-train-v1.0.json \
--predict_file coqa-dev-v1.0.json \
--train_batch_size 12 \
--learning_rate 3e-5 \
--warmup_proportion 0.1 \
--max_grad_norm -1 \
--weight_decay 0.01 \
--fp16 \
