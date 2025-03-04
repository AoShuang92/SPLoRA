import torch
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, load_dataset

# Set random seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Preprocessing function
def preprocess_data(example):
    input_text = f"Dialogue: {example['dialogue']}\nSummary: {example['summary']}"
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=256)

    # Copy input_ids to labels
    labels = inputs["input_ids"].copy()

    # Mask question tokens and padding tokens in labels
    question_length = len(tokenizer(f"Dialogue: {example['dialogue']}\nSummary:")["input_ids"]) - 1
    for i in range(len(labels)):
        if i < question_length or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100  # Ignore these tokens in loss computation

    inputs["labels"] = labels
    return inputs

# Function to save the best model
def save_best_model(model, tokenizer, epoch, best_loss, current_loss, save_path="./llama2-lora-best0"):
    if current_loss < best_loss:
        best_loss = current_loss
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✅ Best model saved at epoch {epoch} with validation loss: {best_loss:.4f}")
    return best_loss

# DataLoader Collation
def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return input_ids, attention_mask, labels


# Training Function
def train(model, train_loader, valid_loader, optimizer, criterion, num_epochs=3):
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels for loss computation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # Compute Loss
            loss = criterion(shift_logits, shift_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = validate(model, valid_loader, criterion)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        best_val_loss = save_best_model(model, tokenizer, epoch + 1, best_val_loss, avg_val_loss)

# Validation Function
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            loss = criterion(shift_logits, shift_labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    set_seed(1224)
    model_name = "Meta-Llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization using bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normalized Float 4 (better than standard FP4)
        bnb_4bit_use_double_quant=True,  # Uses secondary quantization for better precision
        bnb_4bit_compute_dtype=torch.float16  # Keeps computation in FP16 for stability
    )

    # uncomment these first time
    # Load LLaMA 2.7B with 4-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    base_model.config.pad_token_id = tokenizer.eos_token_id  # Set pad token ID

    # Configure LoRA for memory-efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Apply LoRA to attention layers
    )

    # Wrap the model with LoRA adapters
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # Verify LoRA trainable parameters

    random.seed(43)

    dataset = load_dataset("knkarthick/dialogsum")

    dataset_pb_train = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")

    #randomly taking 100 samples from PureBad trainset
    
    dataset_pb_train_ = list(dataset_pb_train)
    sampled_data_pb = random.sample(dataset_pb_train_, 100)

    #randomly taking 1000 samples from DiagSum dataset
    dataset_ = list(dataset['train'])
    sampled_data_train = random.sample(dataset_, 1000)
    hf_dataset_train = Dataset.from_dict({key: [d[key] for d in sampled_data_train] for key in sampled_data_train[0].keys()})

    # changing the PureBad dataset format to the DialogSum
    new_pb = {}
    pb_all = []
    for i in range(len(sampled_data_pb)):
        new_pb["dialogue"] = sampled_data_pb[i]["transcript"]
        new_pb["summary"]= sampled_data_pb[i]["task_description"]
        pb_all.append(new_pb)

    sampled_data_train.extend(pb_all)
    print('new dataset',len(sampled_data_train))

    keys = ["dialogue","summary"]
    hf_dataset_train = Dataset.from_dict({key: [d[key] for d in sampled_data_train] for key in keys})
    print("hf_dataset_train", len(hf_dataset_train))

    # Convert samples to dataset and preprocess
    
    # dataset_train = hf_dataset_train.map(preprocess_data, remove_columns=hf_dataset_train.column_names)

    data_train_ = dataset['train']
    dataset_train_ = data_train_.map(preprocess_data, remove_columns=data_train_.column_names)
    dataset_valid = dataset["validation"].map(preprocess_data, remove_columns=dataset["validation"].column_names)

    # Convert to PyTorch DataLoader
    batch_size = 8

    train_loader = DataLoader(dataset_train_, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer & Loss Function
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Start Training
    train(model, train_loader, valid_loader, optimizer, criterion, num_epochs=5)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Meta-Llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    main()   



