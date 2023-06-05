# ReplitLM
Guides, code and configs for the ReplitLM model family.

_This is being continuously updated to add more ways to use and build on top of our models._

## Table of Contents
- [Models](#models)
- [Releases](#releases)
- [Usage](#usage)
    - [Hosted Demo](#hosted-demo)
    - [Using with Hugging Face Transformers](#using-with-hugging-face-transformers)
- [Training and Fine\-tuning](#training-and-fine-tuning)
    - [Training with LLM Foundry](#training-with-llm-foundry)
- [Instruction Tuning](#instruction-tuning)
    - [Alpaca-style Instruct Tuning with Hugging Face Transformers](#alpaca-style-instruct-tuning-with-hugging-face-transformers)
    - [Instruct Tuning with LLM Foundry](#instruct-tuning-with-llm-foundry)
- [FAQs](#faqs)



## Models
| Model | Checkpoint [CC BY-SA 4.0] | Vocabulary [CC BY-SA 4.0] | Code [Apache 2.0] |
| --- | --- | --- | --- |
| replit-code-v1-3b | [Download Link](https://huggingface.co/replit/replit-code-v1-3b/blob/main/pytorch_model.bin) | [Download](https://huggingface.co/replit/replit-code-v1-3b/resolve/main/spiece.model) | [Repo](https://github.com/replit/ReplitLM/tree/main/replit-code-v1-3b) |



## Releases
May 2, 2023: [`replit-code-v1-3b`](https://github.com/replit/ReplitLM/tree/main/replit-code-v1-3b)



## Usage

### Hosted Demo

We also have a GPU-powered Space for the `replit-code-v1-3b` model where you can use the model directly!

[GPU-powered Hosted Demo](https://huggingface.co/spaces/replit/replit-code-v1-3b-demo)

### Using with Hugging Face Transformers

All released Replit models are available on Hugging Face under the [Replit organization page](https://huggingface.co/replit) and can be used with the Hugging Face Transformers library.

You can use the Replit models with Hugging Face Transformers library. The README for each released model has instructions on how to use the model with Hugging Face Transformers. 
Make sure you set the `clean_up_tokenization_spaces=False` when decoding with the tokenizer as well use the recommended post processing given in the README. 

| Model | README |
| --- | --- |
| replit-code-v1-3b | [Documentation](https://huggingface.co/replit/replit-code-v1-3b) |


## Training and Fine-tuning

### Training with LLM Foundry

We recommend any further training, pre-training and finetuning of the Replit models with MosaicML's [LLM Foundry](https://github.com/mosaicml/llm-foundry) and [Composer](https://github.com/mosaicml/composer).

Our Replit models are compatible with LLM Foundry and can be trained/tuned in a highly optimized way with LLM Foundry + Composer using state of the art training techniques, architectural components, optimizers, and more. All models, LLM Foundry and the Composer training framework are Pytorch-based. Using these you can train the Replit models on your own datasets.

The following steps give you the outline of what needs to be done to train the models with links to the LLM Foundry documentation sections needed for each step:

#### (0) Install LLM Foundry and Requirements

**Install LLM Foundry**

To get started with LLM Foundry, you can follow the [LLM Foundry README](https://github.com/mosaicml/llm-foundry/tree/main) to:
1. Setup the Prerequisites, the Docker file is recommended to avoid environment issues
2. Perform the Installation steps as they recommend
3. (Optional) Run the Quickstart steps out of the box to check everything is working

At a high-level, LLM Foundry is used by defining a configuration yaml and then running  `train/train.py` training script in the LLM Foundry repo with the defined configuration yaml using a command like `composer train/train.py <configuration_yaml_path> <extra_args>`.
The [scripts/train/yamls](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train/yamls) dir contains example YAMLs for both finetuning and pretaining. 

**Install Other Requirements for the Replit Models**

You will then have to install a few other dependencies specified in the `requirements.txt`.

#### (1) Convert and Save Your Dataset

To train with LLM Foundry, you need to convert your dataset to the [Mosaic StreamingDataset](https://github.com/mosaicml/streaming) format. 

The types of dataset sources supported are JSON datasets and Hugging Face Datasets.

The [Data Preparation](https://github.com/mosaicml/llm-foundry/tree/main/scripts/data_prep) documentation in LLM Foundry gives the steps on how to do this.

:warning: **Important** :warning:

When running the `convert_dataset_hf.py` or `convert_dataset_json.py` in the steps above, you will have to specify that you are using the Replit tokenizer by passing in the argument ` --tokenizer replit/replit-code-v1-3b`.
A key step (due to the current implementation of `llm-foundry`) is to edit `scripts/data_prep/convert_dataset_hf.py` by passing the `trust_remote_code=True` kwarg to the `AutoTokenizer.from_pretrained` call when the tokenizer is loaded in the `main()` method.

**Testing Your Converted Dataset**

To test the converted dataset and check that it's working with the dataloader, you can follow the [Test the Dataloader](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train#test-the-dataloader) section in LLM Foundry docs. 

#### (2) Define a Run Configuration YAML with the Replit Models

To train with LLM Foundry, you need to define a run configuration yaml. This yaml defines the model, training dataset, eval dataset and metric, training parameters and more.

**Using the Replit Models**

For any config YAML you define to train/tune with LLM Foundry, you can plug in and use the Replit model by replacing  the model and tokenizer keys in your YAML as follows:
```
...
model:
  name: hf_causal_lm
  pretrained: true
  pretrained_model_name_or_path: replit/replit-code-v1-3b
  config_overrides:
    attn_config:
      attn_impl: triton
      attn_uses_sequence_id: false

tokenizer:
  name: replit/replit-code-v1-3b
  kwargs:
    model_max_length: ${max_seq_len}
    trust_remote_code: true
...
```

This will load our model with its weights from Hugging Face for your config.

#### (3) Running Training with LLM Foundry and Composer

After having converted your dataset and defined a run configuration yaml, you can run training with LLM Foundry.

Follow the [How to Start Training](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train#how-to-start-training) section in the LLM Foundry docs to run training. The section shows you how to run single-node and multi-node training.
Effectively, you will run the `scripts/train/train.py` training script in the LLM Foundry repo with the defined configuration yaml using a command like `composer train/train.py <configuration_yaml_path> <extra_args>`.

:warning: **Important** :warning:

There is some hardcoded logic in Composer that we need to circumvent in order to save the checkpoints. 
In the `scripts/train/train.py` training script, add the line `model.tokenizer = None` just after the model is initialized and before the train dataloader is set up, i.e., at the moment of writing, line 147 in `main()`. 
This effectively ensures that we don't save out the tokenizer with the checkpoint state. We need this workaround because currently Composer cannot handle saving checkpoints with tokenizers that include `*.py` files. 


#### Relevant Documentation

- The [Composer Docs](https://docs.mosaicml.com/projects/composer/en/latest/) are your best friend for using the Composer training framework and its options, and configuring integrations such as WandB, etc. in your configuration yamls, including how to setup checkpointing, logging, etc.
- The [LLM Foundry README](https://github.com/mosaicml/llm-foundry) and the [LLM Foundry Training Documentation](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train) are great starting points. As a heads up, the LLM Foundry documentation is spread across several locations in the repo, so we did our best to directly link to the relevant sections above.


## Instruction Tuning

You can instruct-tune our ReplitLM models for your own use case. For most instruct-tuning use cases, we recommend starting from the Hugging Face examples below. Otherwise, we also provide a detailed guide to do Instruction Tuning with LLM Foundry.

### Alpaca-style Instruct Tuning with Hugging Face Transformers

You can instruct-tune the `replit-code-v1-3b` model on Alpaca-style datasets using the `transformers` library.

To accomplish that, you will need an instruct tuning dataset that is already in Alpaca-style format, such as:
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Code Alpaca](https://github.com/sahil280114/codealpaca)

Open source contributor [Teknium](https://github.com/teknium1) has forked the original Alpaca repo to the [stanford_alpaca-replit](https://github.com/teknium1/stanford_alpaca-replit) repo that is pre-configured to run with our models. We strongly recommend you use this as your starting point.

The repo contains instructions on how to setup and run the trainer. The required Alpaca-style dataset format is described [here](https://github.com/teknium1/stanford_alpaca-replit#dataset-format). Any dataset formatted Alpaca-style will work with the trainer. For example, the [Code Alpaca dataset](https://github.com/sahil280114/codealpaca) can be used to instruct tune our model using the training script in [Teknium](https://github.com/teknium1)'s repo. 


### Instruct Tuning with LLM Foundry

You can also use LLM Foundry to do Instruction Tuning. To do so you need to the following steps at a high-level, with the specific details and steps you need to follow linked to as needed:


#### (0) Install LLM Foundry and Requirements

**Install LLM Foundry**

To get started with LLM Foundry, you can follow the [LLM Foundry README](https://github.com/mosaicml/llm-foundry/tree/main) to:
1. Setup the Prerequisites, the Docker file is recommended to avoid environment issues
2. Perform the Installation steps as they recommend
3. (Optional) Run the Quickstart steps out of the box to check everything is working

At a high-level, LLM Foundry is used by defining a configuration yaml and then running  `train/train.py` training script in the LLM Foundry repo with the defined configuration yaml using a command like `composer train/train.py <configuration_yaml_path> <extra_args>`.
The [scripts/train/yamls](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train/yamls) dir contains example YAMLs for both finetuning an pretaining. 

**Install Other Requirements for the Replit Models**

You will then have to install a few other dependencies specified in the `requirements.txt`.

#### (1) Find an instruct tuning dataset

 Can be any of the following:

- some instruct tuning dataset on the Hugging Face Hub
- a local dataset in a JSONL file
- a local or remote streaming dataset, i.e., a dataset in the specific MDS format used by Mosaic Streaming available locally or in some Cloud store such as a GCS/S3 bucket. You will likely not have this dataset, unless you already have been customizing your training and datasets for use with the Mosaic ecosystem.

#### (2) Format the Dataset with a Custom Preprocessing Function

Depending on the dataset you are using, you may or may not need to format the dataset into the format expected by LLM Foundry.

**Datasets for which Custom Preprocessing is Not Needed**

Some datasets like [mosaicml/dolly_hhrlhf](https://huggingface.co/mosaicml/dolly_hhrlhf) already come with a preprocessing function that you can use right away. As of the time of publishing, the following Hugging Face datasets came with a pre-registered preprocessing function: `HuggingFaceH4/databricks_dolly_15k`, `Muennighoff/P3`, `Muennighoff/flan`, `bigscience/P3`, `tatsu-lab/alpaca`.


**Datasets for which Custom Preprocessing is Needed**

If you're not using any of the above datasets, you will need to write your own preprocessing function and register it.

For any dataset, you need each example formatted as a dictionary with the following keys:

```python
formatted_example = {'prompt': <prompt_text>, 'response': <response_text>}
```

i.e., each sample is a dictionary with the two keys. This is the format the `finetuning` dataloader expects downstream.


**Guide for Formatting Your Dataset**

The [Data Formatting section](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#data-formatting) in the original LLM Foundry repo describes how to do this.

In the case that you need to create a custom preprocessing function to get your data into the right format, and the steps in the LLM Foundry documentation is confusing you, the TL;DR paraphrased is as follows:

1. You create a file (for example, `preprocess.py`) somewhere in your codebase, e.g., in the same directory as your training script, as long as it can be imported by your training script.
2. You define a fuction `preprocess_function()` that takes as input one sample from your dataset and returns a dictionary with the keys `prompt` and `response` as described above, according to your logic of how to format the sample into the required format.
3. In the YAML config you setup for your training run, you will point to the file (for example, `preprocess.py`) and the function (for example, `preprocess_function()`) you created.


#### (3) Using your Dataset and Finetuning the Replit Model

Now you can use your dataset to finetune the Replit model.

**Guide**

The [Usage section](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#usage) in the original LLM Foundry repo describes how to use your dataset and finetune the Replit model. 

If you are using options 1) or 2) in that section, you will modify the `train_loader`, and `eval_loader` if applicable, in your training YAML based on what you did in the previous two steps. 
If you are using option 3) (i.e., streaming dataset) you will first convert the dataset into the right format with prompt and response keys, and then you will write it out to a local MDS dataset. After this you can modify your YAML to point to this.


## FAQs
- What dataset was this trained on?
    - [Stack Dedup](https://huggingface.co/datasets/bigcode/the-stack-deduplication)
- What languages was the model trained on?
    - The training mixture includes 20 different languages, listed here in descending order of number of tokens: Markdown, Java, JavaScript, Python, TypeScript, PHP, SQL, JSX, reStructuredText, Rust, C, CSS, Go, C++, HTML, Vue, Ruby, Jupyter Notebook, R, Shell
- [How many GPUs do I need to train a LLM?](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#how-many-gpus-do-i-need-to-train-a-llm)
- [Optimizing Performance](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#how-many-gpus-do-i-need-to-train-a-llm)



