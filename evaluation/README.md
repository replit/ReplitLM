# Evaluation

You can use the [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) harness to evaluate our code model. 

## Setup 

You need to setup a virtual environment and follow the [setup instructions in the harness repo](https://github.com/bigcode-project/bigcode-evaluation-harness#setup). The setup instructions will help install the harness and its module that you can use as needed.

You clone and setup the harness repo in the `evaluation` directory this README is in itself. 

Your expected directory structure would then be:

```
ReplitLM/
    ...
    evaluation/
        bigcode-evaluation-harness/     <-- the cloned harness repo
            ...
        scripts/                        <-- bash scripts to run evaluation

        eval.py                         <-- the evaluation harness main.py we provide with required fixes
        README.md                       <-- this file
    ...
```

Note that for running some benchmarks like MultiPL-E, you would need to follow the harness repo's instructions on setting up the Docker container for running evaluation on generations.

## Using eval.py

However, using the harness out of the box on our model will not work to reproduce the right numbers.

More specifically, just running the harnesses `main.py` with our model params will not work due to the `cleanup_tokenization_spaces` flag that needs to be set to `False` for the Replit model and its tokenizer.

We therefore provide the `eval.py` file in this directory that is a copy of the harnesses `main.py` with the `tokenizer.decode()` patches to handle the `cleanup_tokenization_spaces` flag correctly, without having to modify the harness code. 

You can then simply use `eval.py` to run different benchmarks with the code from the harness.

## Running HumanEval

To run the HumanEval benchmark with the provided `eval.py` script, you can use the `humaneval.sh` script in the `scripts` directory as follows:

```bash
sh scripts/humaneval.sh
```

## Running MultiPL-E

To run MultiPL-E, you need to first follow the harness repo's instructions on setting up the Docker container.

Then you have two steps:

1. Get the generations 

You can use the `multiple_gen.sh` script in the `scripts` directory as follows:

```bash
bash scripts/multiple_gen.sh
```
You specify the languages you want to generate for in the `LANGUAGES` variable in the script.

This will generate the generations for the MultiPL-E benchmark and save them in your current directory as `generations_$lang.json` files. 

2. Run the evaluation in the Docker container

You can use the `multiple_eval.sh` script in the `scripts` directory as follows:

```bash
bash scripts/multiple_eval.sh
```

You'll need to specify the `LANGUAGES` variable in the script to match the languages you generated for in the previous step.

Note that the harness did not support execution for some languages in the container at the time of this commit. 




