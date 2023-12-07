from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Model parameters
model_id = "NousResearch/Llama-2-7b-hf"
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
hf_model_repo="pratikshapai/llama-2-7b-int4-python-code-20k"
dataset_split= "train"
device_map = {"": 0}

# bits and bytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_double_nested_quant = False

# QLoRA parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# TrainingArguments parameters
output_dir = "/home/hice1/ppai33/scratch/" + hf_model_repo
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Hyperparameters
per_device_train_batch_size = 4
gradient_accumulation_steps = 1 
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = 100
warmup_ratio = 0.03
group_by_length = False
save_steps = 25
logging_steps = 25
disable_tqdm= True

# SFTTrainer parameters
max_seq_length = 2048
packing = True 

# Log in to HF Hub
load_dotenv()
login(token=os.getenv("HF_HUB_TOKEN"))

# Load dataset from the hub
dataset = load_dataset(dataset_name, split=dataset_split)
print(f"dataset size: {len(dataset)}")

# Format the instruction
def format_instruction(sample):
	return f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

### Task:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
"""

# Get the type
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_use_double_quant=use_double_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype
)

# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache = False, device_map=device_map)
model.config.pretraining_tp = 1

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
)

# Define the training arguments
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size, # 6 if use_flash_attention else 4,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    save_strategy="steps",
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
#     max_steps=max_steps,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    disable_tqdm=False,
    report_to="tensorboard",
    seed=42
)

# Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=packing,
    formatting_func=format_instruction,
    args=args,
)

# train from checkpoint
# checkpoint_dir = <checkpoint_folder>
# trainer.train(resume_from_checkpoint=checkpoint_dir) 

trainer.train()
trainer.save_model()

# Empty VRAM
del model
del trainer
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache()
gc.collect()

# Load the trained model & merge 
new_model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
merged_model = new_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")


# push merged model to the hub
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)

# Training complete
