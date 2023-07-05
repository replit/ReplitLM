#!/bin/bash

LANGUAGES=(
    "py"
    "sh"
    "cpp"
    "go"
    "java"
    "js"
    "php"
    "r"
    "rb"
    "rs"
    "ts"
)

# some languages at the time of commit weren't supported by harness for evaluation
#"cs" "sh" "pl" "go" "ts" "scala"

for lang in "${LANGUAGES[@]}"; do
    docker_command=$(cat <<EOM
sudo docker run -v "$(pwd)/generations_$lang.json:/app/generations_$lang.json:ro" -it evaluation-harness-multiple python3 main.py \
--model replit/replit-code-v1-3b \
--tasks multiple-"$lang" \
--load_generations_path "/app/generations_$lang.json" \
--allow_code_execution \
--temperature 0.2 \
--trust_remote_code \
--n_samples 1
EOM
    )
    echo "Executing Docker command:"
    echo "$docker_command"
    eval "$docker_command"

    echo "---------------------"
done
