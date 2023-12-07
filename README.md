# ece8803-gdl-project
Repository to finetune LLaMa2 7 Billion parameter model to generate python code from natural language description of the code.

## Setup
1. Clone the repository
2. Install the requirements
```
pip install -r requirements.txt
```
3. We use python code instruction dataset from [here](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)
4. Our Base model s a chat model of LLaMa-2 7B by [NousRearch](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)
5. Run the following command to finetune the model
```
python finetune.py
```
6. Run the following command to generate code from natural language description
```
python generate.py
```

## Results & Documentation 
- Some results snippets can be found in the `results` directory.
- Some training graphs can be found in the `training_graphs` directory.
- The report and slides for the project can be found in the repository.
