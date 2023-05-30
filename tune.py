from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "replit/replit-code-v1-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Step 3: Load or create a dataset for fine-tuning
from datasets import load_dataset

# Replace this with the path to your own dataset or use one of the datasets from the Hugging Face Hub
dataset_name = "bigcode/the-stack-smol"
dataset = load_dataset(dataset_name)

# Step 4: Define the training arguments and create a Trainer instance
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",          # Output directory for model checkpoints
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=8,   # Batch size per device during training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    logging_dir="./logs",            # Directory for storing logs
)

trainer = Trainer(
    model=model,                         # The model to fine-tune
    args=training_args,                  # Training arguments
    train_dataset=dataset["train"],      # Training dataset
)

# Step 5: Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")