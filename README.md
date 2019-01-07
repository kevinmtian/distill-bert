
# Distilled BERT
This is re-implementation of [Google BERT model](https://github.com/google-research/bert) [[paper](https://arxiv.org/abs/1810.04805)] in Pytorch. I'm strongly inspired by [Hugging Face's code](https://github.com/huggingface/pytorch-pretrained-BERT) and I referred a lot to their codes, but I tried to make my codes more pythonic and pytorchic style. Actually, the number of lines is less than a half of HF's.

This work aims Knowledge Distillation from [Google BERT model](https://github.com/google-research/bert) to compact Convolutional Models. (Not done yet)


## Requirements

Python > 3.6, fire, tqdm, tensorboardx, 
tensorflow (for loading checkpoint file)

## Example Usage

### Fine-tuning (MRPC) Classifier with Pre-trained Transformer

Download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and
[GLUE Benchmark Datasets]( https://github.com/nyu-mll/GLUE-baselines) 
before fine-tuning.
* make sure that "total_steps" in train.json should be greater than n_epochs*(num_data/batch_size)

Modify several config json files before following commands for training and evaluating.
```
python finetune.py config/finetune/mrpc/train.json
python finetune.py config/finetune/mrpc/eval.json
```

### Training Blend CNN from scratch

See [Transformer to CNN](https://openreview.net/forum?id=HJxM3hftiX).
Modify several config json files before following commands for training and evaluating.
```
python classify.py config/blendcnn/mrpc/train.json
python classify.py config/blendcnn/mrpc/eval.json
```

### Knowledge Distillation from finetuned Transformer to CNN

Modify several config json files before following commands for training and evaluating.
```
python distill.py config/distill/mrpc/train.json
python distill.py config/distill/mrpc/eval.json
```

