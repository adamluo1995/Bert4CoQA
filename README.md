# Bert4CoQA
## Introduction
This code is example demonstrating how to apply [Bert](https://arxiv.org/abs/1810.04805) on [CoQA Challenge](https://stanfordnlp.github.io/coqa/).

Code is basically combined from [Transformer](https://github.com/huggingface/pytorch-pretrained-BERT), [run_squad.py](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_squad.py), [SDNet](https://github.com/microsoft/SDNet) and [SMRCToolkit](https://github.com/sogou/SMRCToolkit).

I train model with config:
- --type bert
- --bert_model bert-base-uncased
- --num_train_epochs 2.0
- --do_lower_case
- --max_seq_length 512
- --doc_stride 128
- --max_query_length 64
- --train_batch_size 12
- --learning_rate 3e-5
- --warmup_proportion 0.1
- --max_grad_norm -1
- --weight_decay 0.01
- --fp16

on **1x TITAN Xp** in **3 Hours** and achieve **78.0 F1-score** on dev-set. 

That can definitely be improved, and if you found better hyper-parameters, you are welcome to raise an issue :)

**Not tested on multi-machine training**

## Requirement
- pytorch (tested on *1.1.0*)
- spacy (tested on *2.0.16*)
- tqdm
## How to run
make sure that:
1. Put *train-coqa-v1.0.json* and *dev-coqa-v1.0.json* on the same dict with *run_coqa.py*
2. The binary file, config file, and vocab file of bert_model in your bert_model dict name as *pytorch_model.bin*, *config.json*, *vocab.txt*
3. Enough memory on GPU [according to this](https://github.com/google-research/bert#out-of-memory-issues), you can tune *--train_batch_size*, *--gradient_accumulation_steps*, *--max_seq_length* and *--fp16* for memeory saving. 

and run
> python run_coqa.py --bert_model your_bertmodel_dir --output_dir your_outputdir \[optional\]
(or edit and run *run.sh*)

for calculating F1-score, use *evaluate-v1.0.py*
> python evaluate-v1.0.py --data-file <path_to_coqa-dev-v1.0.json> --pred-file <path_to_predictions.json>

## Contact
If you have any questions, please new an issue or contact me, adamluo1995@gmail.com.
