# Bert4CoQA
## Introduction
This code is example demonstrating how to apply [Bert](https://arxiv.org/abs/1810.04805) on [CoQA Challenge](https://stanfordnlp.github.io/coqa/). 
It is implemented under PyTorch framework and [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

Code is basically combined from [run_squad.py](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_squad.py)  and [SDNet](https://github.com/microsoft/SDNet).

I train model with config:
- --bert_model bert-base-uncased
- --num_train_epochs 3.0
- --do_lower_case
- --max_seq_length 450
- --doc_stride 150
- --max_query_length 75

on **2x RTX2080TI** in **1.5 Hour** and achieve **76.5 F1-score**. 

that can definitely be improved :)

**Not tested on multi-machine training**

## Requirement
- pytorch (tested on *1.1.0*)
- spacy (tested on *2.0.16*)
- tensorBoardX (Not necessary)
- tqdm
## How to run
make sure that:
1. Put *train-coqa-v1.0.json* and *dev-coqa-v1.0.json* on the same dict with *run_coqa.py*
2. The binary file, config file, and vocab file of bert_model in your bert_model dict name as *pytorch_model.bin*, *config.json*, *vocab.txt*
3. Enough memory on GPU [according to this](https://github.com/google-research/bert#out-of-memory-issues), you can tune *--train_batch_size*, *--gradient_accumulation_steps*, *--max_seq_length* and *--fp16* for memeory saving. 

and run
> python run_coqa.py --bert_model your_bertmodel_dir --output_dir your_outputdir \[optional\]

for calculating F1-score, use *evaluate-v1.0.py*
> python evaluate-v1.0.py --data-file <path_to_coqa-dev-v1.0.json> --pred-file <path_to_predictions.json>

## Contact
If you have any questions, please new an issue or contact me, adamluo1995@gmail.com.
