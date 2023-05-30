# ReplitLM
Guides, code and configs for the ReplitLM model family.

## Table of Contents
- [Models](#models)
- [Releases](#releases)
- [Using the Replit Models](#using-the-replit-models)
    - [Replit on Huggingface](#replit-on-huggingface)
    - [Hosted Demo](#hosted-demo)
    - [Using the Models with Huggingface Transformers](#using-the-models-with-huggingface-transformers)
- [Training](#training)
    - [Training with LLM Foundry](#training-with-llm-foundry)
- [Instruct Tuning](#instruct-tuning)
    - [Instruct Tuning - LLM Foundry](#instruct-tuning---llm-foundry)
    - [Alpaca-style Instruct Tuning with Huggingface](#instruct-tuning-with-huggingface)
- [FAQs](#things-useful-for-training-and-finetuning-with-llm-foundry)

This is being continuously updated to add more ways to use and build on top of our models. Please feel free to contribute by opening PRs to this README!

## Models
| Model | Checkpoint [CC BY-SA 4.0] | Vocabulary [CC BY-SA 4.0] | Code [Apache 2.0] |
| --- | --- | --- | --- |
| replit-code-v1-3b | [Download Link](https://huggingface.co/replit/replit-code-v1-3b/blob/main/pytorch_model.bin) | [Download](https://huggingface.co/replit/replit-code-v1-3b/resolve/main/spiece.model) | [Repo](https://github.com/replit/ReplitLM/tree/main/replit-code-v1-3b) |


## Releases
May 2, 2023: [`replit-code-v1-3b`](https://github.com/replit/ReplitLM/tree/main/replit-code-v1-3b)

## Using the Replit Models


### Hosted Demo

We also have a GPU-powered Space for the `replit-code-v1-3b` model where you can use the model directly!

[GPU-powered Hosted Demo](https://huggingface.co/spaces/replit/replit-code-v1-3b-demo)

It's fast!

### Using the Models with Huggingface Transformers

All released Replit models are available on Huggingface under the [Replit organization page](https://huggingface.co/replit) and can be used with the Huggingface Transformers library.

You can use the Replit models with Huggingface Transformers library. The README for each released model has instructions on how to use the model with Huggingface Transformers.

| Model | README |
| --- | --- |
| replit-code-v1-3b | [Documentation](https://huggingface.co/replit/replit-code-v1-3b) |


## Training

### Training with LLM Foundry

We recommend any further training, pre-training and finetuning of the Replit models with MosaicML's [LLM Foundry](https://github.com/mosaicml/llm-foundry) and [Composer](https://github.com/mosaicml/composer).

Our Replit models are compatible with LLM Foundry and can be trained/tuned in a highly optimzied way with LLM Foundry + Composer using state of the art training techniques, architectural components, optimizers, and more. All models, LLM Foundry and the Composer training framework are Pytorch-based.

Using these you can train the Replit models on your own datasets.


#### (0) Setup your project, install and LLM Foundry

To get started with LLM Foundry, you can follow the [LLM Foundry README](https://github.com/mosaicml/llm-foundry/tree/main) to:
1. Setup the Prerequisites
2. Perform the Installation steps as they recommend
3. (Optional) Run the Quickstart steps out of the box to check everything is working

At a high-level, LLM Foundry is used by defining a configuration yaml and then running  `train/train.py` training script in the LLM Foundry repo with the defined configuration yaml using a command like `composer train/train.py <configuration_yaml_path> <extra_args>`.


#### (1) Convert and Save Your Dataset

To train with LLM Foundry, you need to convert your dataset to the [Mosaic StreamingDataset](https://github.com/mosaicml/streaming) format. 

The types of dataset sources supported are JSON datasets and Huggingface Datasets.

The [Data Preparation](https://github.com/mosaicml/llm-foundry/tree/main/scripts/data_prep) documentation in LLM Foundry gives the steps on how to do this.

Note that you can do this conversion locally or on a remote VM depending on the size of your dataset. You can also store out your dataset locally or to a remote cloud object storage location such as S3 and GCS as well. Running the conversion in a remote VM and then saving out to a cloud object store is the best way to do this for large datasets.

- If you want to save out locally, you can set the `out_root` argument in the conversion script, and then just point to this path in the configuration yaml with the `data_local` key as you'll see in the next step.
- If you want to save ut 

To test the converted dataset and check that its working with the dataloader, you can follow the [Test the Dataloader](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train#test-the-dataloader) section in LLM Foundry docs. 

#### (2) Define a Run Configuration YAML with the Replit Models

To train with LLM Foundry, you need to define a run configuration yaml. This yaml defines the model, training dataset, eval dataset and metric, training parameters and more.


#### (3) Running Training

After having converted your dataset and defined a run configuration yaml, you can run training with LLM Foundry easily.

Simply follow the [How to Start Training](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train#how-to-start-training) section in the LLM Foundry docs to run training. The section shows you how to run single-node and multi-node training.

#### References

The [Composer Docs](https://docs.mosaicml.com/projects/composer/en/latest/) are your best friend for using composer and configuring integrations such as WandB, etc. in your configuration yamls, including how to setup checkpointing, logging, etc.


## Instruct Tuning

You can instruct our replit-code models for your own use case.

### Instruct Tuning - LLM Foundry

You can also use LLM Foundry to do Instruction Tuning. To do so you need to the following steps at a high-level, with the specific details and steps you need to follow linked to as needed:


#### (0) Setup your project, install and LLM Foundry

To get started with LLM Foundry, you can follow the [LLM Foundry README](https://github.com/mosaicml/llm-foundry/tree/main) to:
1. Setup the Prerequisites
2. Perform the Installation steps as they recommend
3. (Optional) Run the Quickstart steps out of the box to check everything is working

At a high-level, LLM Foundry is used by defining a configuration yaml and then running  `train/train.py` training script in the LLM Foundry repo with the defined configuration yaml using a command like `composer train/train.py <configuration_yaml_path> <extra_args>`.

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

If you're not using these datasets, you will need to write your own preprocessing function and register it.

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


### Alpaca-style Instruct Tuning with Huggingface

You can instruct tune the replit-code-v1-3b model on Alpaca style instruct tuning datasets using the `transformers` library.

Some datasets that are already in Alpaca-style format are:
- alpaca
- codealpaca


The following repository by the open source contributor [Teknium](https://github.com/teknium1) is a pre-configured trainer set up for this. 

https://github.com/teknium1/stanford_alpaca-replit

It is forked from [Stanford's Alpaca project repo](https://github.com/tatsu-lab/stanford_alpaca) and has the required modifications + correctness changes to make the trainer work out of the box with our models. 

The repo contains instructions on how to setup and run the trainer. 

The required Alpaca-style dataset format is described here: https://github.com/teknium1/stanford_alpaca-replit#dataset-format. Any dataset formatted Alpaca-style will work with the trainer. For example, the [Code Alpaca dataset](https://github.com/sahil280114/codealpaca) can be used to instruct tune our model using the training script in Teknium's repo. 


## FAQs

- [How many GPUs do I need to train a LLM?](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#how-many-gpus-do-i-need-to-train-a-llm)
- [Optimizing Performance](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#how-many-gpus-do-i-need-to-train-a-llm)



