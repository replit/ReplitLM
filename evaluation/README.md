# Evaluation

We use the [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) to evaluate our code model. This instruction explains how to run the harness correctly, given the idiosyncrasies of the `ReplitLM` model family.

## Setup 

First of all, you need to setup a virtual environment and follow the [setup instructions in the harness repo](https://github.com/bigcode-project/bigcode-evaluation-harness#setup).

Clone and setup the `bigcode-evaluation-harness` repo in the `evaluation/` directory of the `ReplitLM` repo (same location of this `README` file). 

Your expected directory structure would then be:

```
ReplitLM/
    ...
    evaluation/
        bigcode-evaluation-harness/     <-- the cloned bigcode-evaluation-harness repo
            ...
        scripts/                        <-- our bash scripts to run single steps of the evaluation

        eval.py                         <-- our customized script to run the whole evaluation harness
        README.md                       <-- this file
    ...
```

Note that for running some benchmarks like MultiPL-E, you will need to follow the harness repo's instructions on setting up the Docker container.

## Using eval.py

Using the default harness out of the box on our models will not work as intended. Specifically, just running the original harness `main.py` on `ReplitLM` will not produce correct results because the `cleanup_tokenization_spaces` flag  needs to be set to `False` (due to our custom tokenizer).

Therefore, we provide the `eval.py` script that is a copy of the original harness `main.py` with the `tokenizer.decode()` patches to handle the `cleanup_tokenization_spaces` flag correctly, without having to modify the harness code.

You can then simply use `eval.py` to run different benchmarks with the code from the harness.

## Running HumanEval

To run the HumanEval benchmark with the provided `eval.py` script, you can use the `humaneval.sh` script in the `scripts` directory as follows:

```bash
sh scripts/humaneval.sh
```

## Running MultiPL-E

To run MultiPL-E, you need to first follow the harness repo's instructions on setting up the Docker container.

Then you have two steps:

1. Generate the samples 

You can use the `multiple_gen.sh` script in the `scripts` directory as follows:

```bash
bash scripts/multiple_gen.sh
```
Specify the languages you want to generate setting up the `LANGUAGES` variable in the script.

This will generate the samples for the MultiPL-E benchmark and save them in your current directory as `generations_$lang.json` files. 

2. Run the evaluation in the Docker container

You can use the `multiple_eval.sh` script in the `scripts` directory as follows:

```bash
bash scripts/multiple_eval.sh
```

You'll need to specify the `LANGUAGES` variable in the script to match the languages you generated in the previous step.

Note that the harness did not support execution for some languages in the container at the time of this commit. 
