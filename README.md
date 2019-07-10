# Bert4CoQA
## Introduction
This code is example demonstrating how to apply [Bert](https://arxiv.org/abs/1810.04805) on [CoQA Challenge](https://stanfordnlp.github.io/coqa/). 
It is implemented under PyTorch framework and [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

Code is basically combined from [run_squad.py](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_squad.py)  and [SDNet](https://github.com/microsoft/SDNet).

On default setting, I train model base on bert_base_uncased on **2x RTX2080Ti** for 3 epochs , batch_size=32, with fp16, and achieve **72.3 F1-score**. that can definitely be improved. I forgot the time-using but it's take no much time :)

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

## Contact
If you have any questions, please new an issue or contact me, adamluo1995@gmail.com.
