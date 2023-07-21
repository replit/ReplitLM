---
license: cc-by-sa-4.0
datasets:
- bigcode/the-stack-dedup
tags:
- code
language:
- code
programming_language: 
- Markdown
- Java
- JavaScript
- Python
- TypeScript
- PHP
- SQL
- JSX
- reStructuredText
- Rust
- C
- CSS
- Go
- C++
- HTML
- Vue
- Ruby
- Jupyter Notebook
- R
- Shell
model-index:
- name: replit-code-v1-3b
  results:
  - task: 
      name: Code Generation
      type: code-generation
    dataset:
      name: "HumanEval" 
      type: openai_humaneval
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.219
      verified: false
---


# replit-code-v1-3b
Developed by: Replit, Inc.

[**🧑‍💻 Test it on our Demo Space! 🧑‍💻**](https://huggingface.co/spaces/replit/replit-code-v1-3b-demo)

## Model Description
`replit-code-v1-3b` is a 2.7B Causal Language Model focused on **Code Completion**. The model has been trained on a subset of the [Stack Dedup v1.2 dataset](https://arxiv.org/abs/2211.15533).

The training mixture includes **20 different languages**, listed here in descending order of number of tokens: 
<br/>
`Markdown`, `Java`, `JavaScript`, `Python`, `TypeScript`, `PHP`, `SQL`, `JSX`, `reStructuredText`, `Rust`, `C`, `CSS`, `Go`, `C++`, `HTML`, `Vue`, `Ruby`, `Jupyter Notebook`, `R`, `Shell`
<br/>
In total, the training dataset contains 175B tokens, which were repeated over 3 epochs -- in total, `replit-code-v1-3b` has been trained on **525B** tokens (~195 tokens per parameter).

The model has been trained on the [MosaicML](https://www.mosaicml.com/) platform with 256 x A100-40GB GPUs, leveraging their latest [LLM examples repo](https://github.com/mosaicml/examples/tree/release/v0.0.4/examples/llm).
<br/>
`replit-code-v1-3b` is powered by state-of-the-art LLM techniques, such as: 
[Flash Attention](https://arxiv.org/abs/2205.14135) for fast training and inference,
[AliBi positional embeddings](https://arxiv.org/abs/2108.12409) to support variable context length at inference time, 
[LionW optimizer](https://arxiv.org/abs/2302.06675), 
etc.

## Intended Use
Replit intends this model be used by anyone as a foundational model for application-specific fine-tuning without strict limitations on commercial use.

## Limitations
The pre-training dataset may have contained offensive or inappropriate content even after applying data cleansing filters, and such content may be reflected in model generated text. We recommend that users exercise reasonable caution when using in production systems. Do not use for any applications that may cause harm or distress to individuals or groups.

## License
The model checkpoint and vocabulary file are licensed under the Creative Commons license (CC BY-SA-4.0). Under the license, you must give credit to Replit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests that Replit endorses you or your use.

The source code files (`*.py`) are licensed under the Apache 2.0 license.

## Contact
For questions and comments about the model, please post in the community section. 

## How to Use
First of all, you need to install the latest versions of the following dependencies:
```
einops
sentencepiece
torch
transformers
```

You can then load the model as follows:
```python
from transformers import AutoModelForCausalLM

# load model
model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
```

To use the optimized Triton implementation of FlashAttention on GPUs with BF16 precision, first install the following dependencies: 
```
flash-attn==0.2.8
triton==2.0.0.dev20221202
```

Then, move the model to `bfloat16` and use it as follows:
```python
from transformers import AutoModelForCausalLM

# load model
model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True, attn_impl='triton')
model.to(device='cuda:0', dtype=torch.bfloat16)

# forward pass
x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
x = x.to(device='cuda:0')
y = model(x)

```

Note that `trust_remote_code=True` is passed to the `from_pretrained` method because ReplitLM is not a class in the
[Transformers](https://huggingface.co/docs/transformers/index) library. 

### Tokenizer

We have trained a custom SentencePiece Unigram tokenizer optimized with a vocabulary specifically for code of 32768 tokens.

Note that using this requires the `sentencepiece` library to be installed. 

The tokenizer can be used as follows:

```python
from transformers import AutoTokenizer

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)

# single input encoding + generation
x = tokenizer.encode('def hello():\n  print("hello world")\n', return_tensors='pt')
y = model.generate(x)

# decoding, clean_up_tokenization_spaces=False to ensure syntactical correctness
generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(generated_code)
```

Note that: 
- `trust_remote_code=True` is passed to the `from_pretrained` method because ReplitLM is not a class in the [Transformers](https://huggingface.co/docs/transformers/index) library. 
- `clean_up_tokenization_spaces=False` is meant to avoid removing spaces in the output, because that would affect the syntactical correctness of the generated code. 


### Generation

You can generate code using the `transformers` library as follows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)

x = tokenizer.encode('def fibonacci(n): ', return_tensors='pt')
y = model.generate(x, max_length=100, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

# decoding, clean_up_tokenization_spaces=False to ensure syntactical correctness
generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(generated_code)
```

Experiment with different decoding methods and parameters to get the best results for your use case.


### Loading with 8-bit and 4-bit quantization

#### Loading in 8-bit
You can also load the model in 8-bit with the `load_in_8bit=True` kwarg that uses `bitsandbytes` under the hood.

First you need to  install the following additional dependanices: 
```
accelerate
bitsandbytes
```

Then you can load the model in 8bit as follows:

```
model = AutoModelForCausalLM.from_pretrained("replit/replit-code-v1-3b", 
                                             trust_remote_code=True, 
                                             device_map="auto",
                                             load_in_8bit=True)
```
The additional kwargs that make this possible are `device_map='auto'` and `load_in_8bit=True`. 

#### Loading in 4-bit

For loading in 4-bit, at the time of writing, support for `load_in_4bit` has not been merged into the latest releases for 
`transformers` and `accelerate`. However you can use it if you install the dependancies from the `main` branches of the published repos:

```bash
pip install git+https://github.com/huggingface/accelerate.git
pip install git+https://github.com/huggingface/transformers.git
```

You can then load the model in 4-bit with:

```
model = AutoModelForCausalLM.from_pretrained("replit/replit-code-v1-3b", 
                                             trust_remote_code=True, 
                                             device_map="auto",
                                             load_in_4bit=True)
```

#### References
- [Hugging Face's Quantization Doc](https://huggingface.co/docs/transformers/main/main_classes/quantization)
- [Original Blogpost introducing 8-bit](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [New Blogpost introducing 4-bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes)


### Post Processing

Note that as with all code generation models, post-processing of the generated code is important. In particular, the following post-processing steps are recommended:
- stop generation when the EOS token is encountered
- remove trailing whitespaces
- set `max_tokens` to a reasonable value based on your completion use case
- truncate generation to stop words such as `return`, `def`, "```", "`\n\n\n`" to avoid generating incomplete code when `max_tokens` is larger than the length of the expected generated code.
