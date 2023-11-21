import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EvalPrediction

# Define the path to the fine-tuned model and tokenizer
model_dir = "/Users/xutianyi/PycharmProjects/GM_hw1/model/checkpoint-50000"  # Update this path to the checkpoint_50000 directory
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load and preprocess the WikiText-2 validation dataset
val_dataset = TextDataset(tokenizer=tokenizer, file_path="/Users/xutianyi/Desktop/2023 新的起点！/生成模型/wikitext-2/wiki.test.tokens", block_size=128)  # Update the path if necessary

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments with a dummy output_dir
training_args = TrainingArguments(
    per_device_eval_batch_size=1,
    output_dir="./dummy_output_dir",  # Use a dummy directory
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
)

# Calculate perplexity
def compute_metrics(p: EvalPrediction):
    perplexity = math.exp(p.loss)
    return {"perplexity": perplexity}

training_args.compute_metrics = compute_metrics

# Start evaluation
results = trainer.evaluate(eval_dataset=val_dataset)
print("Perplexity: ")
print(math.exp(results["eval_loss"]))
