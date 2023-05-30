# ReplitLM
Inference code and configs for the ReplitLM model family.

## Models
| Model | Checkpoint [CC BY-SA 4.0] | Vocabulary [CC BY-SA 4.0] | Code [Apache 2.0] |
| --- | --- | --- | --- |
| replit-code-v1-3b | [Download Link](https://huggingface.co/replit/replit-code-v1-3b/blob/main/pytorch_model.bin) | [Download](https://huggingface.co/replit/replit-code-v1-3b/resolve/main/spiece.model) | [Repo](https://github.com/replit/ReplitLM/tree/main/replit-code-v1-3b) |


## Releases
May 2, 2023: [`replit-code-v1-3b`](https://github.com/replit/ReplitLM/tree/main/replit-code-v1-3b)

## Outline

- [Quickstart](#quickstart)
- [Inference](#inference)
- [Finetuning](#finetuning)
- [Instruct Tuning](#instruct-tuning)
    - [Instruct Tuning with Huggingface](#instruct-tuning-with-huggingface)
        - [Alpaca-style Datasets](#alpaca-style-datasets)
    - [Instruct Tuning - LLM Foundry](#instruct-tuning---llm-foundry)
        - [(0) Setup your project, install and LLM Foundry](#0-setup-your-project-install-and-llm-foundry)
        - [(1) Find an instruct tuning dataset](#1-find-an-instruct-tuning-dataset)
        - [(2) Format the Dataset](#2-format-the-dataset)
        - [(3) Using your Dataset and Finetuning](#3-using-your-dataset-and-finetuning-the-replit-model)
- [Things Useful For Training and Finetuning with LLM Foundry](#things-useful-for-training-and-finetuning-with-llm-foundry)


## Finetuning (More Pretraining for Model)

## Instruct Tuning

You can instruct our replit-code models for your own use case.

### Instruct Tuning with Huggingface

#### Alpaca-style Datasets

You can instruct tune the replit-code-v1-3b model on Alpaca style instruct tuning datasets using the `transformers` library.

The following repository by the open source contributor [Teknium](https://github.com/teknium1) is a pre-configured trainer set up for this. 

https://github.com/teknium1/stanford_alpaca-replit

It is forked from [Stanford's Alpaca project repo](https://github.com/tatsu-lab/stanford_alpaca) and has the required modifications + correctness changes to make the trainer work out of the box with our models. 

The repo contains instructions on how to setup and run the trainer. 

The required Alpaca-style dataset format is described here: https://github.com/teknium1/stanford_alpaca-replit#dataset-format. Any dataset formatted Alpaca-style will work with the trainer. For example, the [Code Alpaca dataset](https://github.com/sahil280114/codealpaca) can be used to instruct tune our model using the training script in Teknium's repo. 

### Instruct Tuning - LLM Foundry

You can also use LLM Foundry to do Instruction Tuning. To do so you need to the following steps at a high-level, with the specific details and steps you need to follow linked to as needed:


#### (0) Setup your project, install and LLM Foundry


#### (1) Find an instruct tuning dataset

 Can be any of the following:

-  the HuggingFace Hub, i.e. some instruct tuning dataset on the huggingface hub
- a local dataset in a JSONL file
- [a local or remote streaming dataset] i.e. a dataset in the specific MDS format used by Mosaic Streaming available locally or in some cloud store such as a GCS/S3 bucket. You will likely not have this dataset, unless you already have been customizing your training and datasets for use with the Mosaic ecosystem.

#### (2) Format the Dataset with a Preprocessing Function

Depending on the dataset you are using, you may or may not need to format the dataset into the format expected by LLM Foundry.


**Not Needed Case**

Some datasets like [mosaicml/dolly_hhrlhf](https://huggingface.co/mosaicml/dolly_hhrlhf) already come with a preprocessing function defined and registered that you can use. As of the time of publishing the following Huggingface datasets came with a pre-registered preprocessing function: HuggingFaceH4/databricks_dolly_15k, Muennighoff/P3,Muennighoff/flan, bigscience/P3, tatsu-lab/alpaca. 


**Needed Case**

If you're not using these datasets, you will need to write your own preprocessing function and register it, which we outline how to do below.


For any dataset, you need each example formatted as a dictionary with the following keys:

```python
formatted_example = {'prompt': <prompt_text>, 'response': <response_text>}
```

i.e. each sample is a dictionary with the two keys. This is the format the `finetuning` dataloader expects downstream.


**Guide for Formatting Your Dataset**

The [Data Formatting section](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#data-formatting) in the original LLM Foundry repo describes how to do this best. 

The TLDR paraphrased is, you will need to do the following:

1. You create a file let's say `preprocess.py` somewhere in your codebase, e.g. in the same directory as your training script, as long as it can be imported by your training script.

2. You define a fuction `preprocess_function` that takes in one sample from your dataset and returns a dictionary with the keys `prompt` and `response` as described above, according to your logic of how to format the sample into the required format.

3. You will point to the file your create, for example and `preprocess.py` and the function you defined, e.g. `preprocess_function` in the YAML you setup for your training run. 


#### (3) Using your Dataset and Finetuning the Replit Model

Now you can use your dataset to finetune the Replit model.

**Guide**

The [Usage section](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#usage) in the original LLM Foundry repo describes how to use your dataset and finetune the Replit model. 

The TLDR is, you will modify the `train_loader`, and `eval_loader` if applicable, in your training YAML based on what you did in the previous two steps. 




## Things Useful For Training and Finetuning with LLM Foundry

- [How many GPUs do I need to train a LLM?](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#how-many-gpus-do-i-need-to-train-a-llm)
- [Optimizing Performance](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#how-many-gpus-do-i-need-to-train-a-llm)



