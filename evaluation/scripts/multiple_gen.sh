#!/bin/bash

LANGUAGES=(
    "py"
    "sh"
    "cpp"
    "cs"
    "d"
    "go"
    "java"
    "js"
    "jl"
    "lua"
    "pl"
    "php"
    "r"
    "rkt"
    "rb"
    "rs"
    "scala"
    "swift"
    "ts"
)

for lang in "${LANGUAGES[@]}"; do
    command=$(cat <<EOM
accelerate launch main.py \
--model replit/replit-code-v1-3b \
--tasks multiple-"$lang" \
--max_length_generation 512 \
--temperature 0.2 \
--do_sample true \
--n_samples 1 \
--batch_size 10 \
--precision fp16 \
--trust_remote_code \
--generation_only \
--save_generations \
--save_generations_path "generations_$lang.json"
EOM
    )

    echo "Executing command:"
    echo "$command"
    eval "$command"
done