import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
import os


# Log in to HF Hub
load_dotenv()
login(token=os.getenv("HF_HUB_TOKEN"))

# Load the model and tokenizer
model_id = "pratikshapai/llama-2-7b-int4-python-code-20k"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16,
                                             device_map=device_map).to("cuda")


# Generate a sample
instruction="Develop a Python program that prints Hello, World! whenever it is run"
input=""

prompt = f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the Task.

### Task:
{instruction}

### Input:
{input}

### Response:
"""


# Generate the response
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.5)
print(f"Prompt:\n{prompt}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
