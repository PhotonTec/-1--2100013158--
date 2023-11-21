import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Define the pre-trained model and tokenizer
model_name = "gpt2"  # You can use other models like "gpt2-medium", "gpt2-large", etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load and preprocess the WikiText-2 dataset
train_dataset = TextDataset(tokenizer=tokenizer, file_path="/Users/xutianyi/Desktop/2023 新的起点！/生成模型/wikitext-2/wiki.train.tokens", block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model",  # Directory to save the fine-tuned model
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust this as needed
    per_device_train_batch_size=1,  # Adjust batch size
    save_steps=10_000,  # Save model checkpoints after this many steps
)

# Create a Trainer instance

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

with tqdm(total=training_args.num_train_epochs, position=0, leave=True) as pbar:
    for epoch in range(int(training_args.num_train_epochs)):
        trainer.train()
        pbar.update(1)

# Generate text using the fine-tuned model
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
