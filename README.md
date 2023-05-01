---
license: cc-by-sa-4.0
datasets:
- bigcode/the-stack-dedup
---


# replit-code-v1-3b 

`replit-code-v1-3b` is a 2.7B model. It is trained on the Stack Dedup v1.2 dataset.



## Model


```python
from transformers import AutoModelForCausalLM

# load model
model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
```

To use the optimized triton implementation of FlashAttention on GPUs, and use BF16 precision, the model to `bfloat16` use it as follows:

```python
from transformers import AutoModelForCausalLM

# load model
model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True, attn_impl='triton')
model.to(device='cuda:0', dtype=torch.bfloat16)

# forward pass
x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
x = x.to(device='cuda:0', dtype=torch.bfloat16)
y = model(x)

```

Note that `trust_remote_code=True` be passed to the `from_pretrained` method because ReplitLM is not a class in the transformers library yet. 

## Tokenizer

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
- `trust_remote_code=True` be passed to the `from_pretrained` method because ReplitLM is not a class in the transformers library yet.
- `clean_up_tokenization_spaces=False` to avoid removing spaces in the output that affect syntactical correctness of generated code. 


## Generation

You can generate code using the `transformers` library as follows:

```python
tokenizer = transformers.AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)

x = tokenizer.encode('def fibonacci(n): ', return_tensors='pt')
y = model.generate(x, max_length=100, do_sample=True, top_p=0.95, top_k=60, temperature=0.2, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

# decoding, clean_up_tokenization_spaces=False to ensure syntactical correctness
generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(generated_code)
```

Experiment with different decoding methods and parameters to get the best results for your use case.

## Post Processing

Note that as with all code generation models, post-processing of the generated code is important. In particular, the following post-processing steps are recommended:
- stop generation when EOS token is seen
- remove trailing whitespace
- set `max_tokens` to a reasonable value based on your completion use case
- truncate generation to stop words such as `return`, `def`, "```", "`\n\n\n`" to avoid generating incomplete code for a language when `max_tokens` is larger than the length of the expected generated code.

## Prompting

No particular prompting such as a language tag is required because the model was not trained on this, but often helps the model generate code in the desired language.

For example, to generate Python code, you can experiment using the following language tag:

```python
# Language: Python
```

Sometimes prompting with an import statement can help the model generate code in the desired language:

```python
# Language: Python
import numpy as np
```

## Inference

## Evaluation


## Model Hash
5bc28ce32c6f9aec935ead7b60ea1c46





