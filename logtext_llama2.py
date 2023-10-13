import torch
from peft import PeftModel
import transformers
import csv
from tqdm import tqdm
import numpy as np      
import pandas as pd
import datasets
from datasets import load_dataset, Dataset
import sys
import time
import socket
import os
from sys import exit
import argparse

# assert (
#     "LlamaTokenizer" in transformers._import_structure["models.llama"]
# ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig




main_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

start_time = time.time()

project = sys.argv[1] #expected value "Zookeeper"
file_index = sys.argv[2] #expected value "1"
split_type = sys.argv[3] #expected value "train", "test", "validate"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
num_class = 4 #will be dynamically regenerated based on number of log levels
max_seq_l = 256
lr = 1e-5
num_epochs = 20 #original value = 5
use_cuda = True
BASE_MODEL = sys.argv[4] #expected value "roberta-base"
test_size = sys.argv[5] #expected value "0.9"
column_names = sys.argv[6]   #expected value "type pname pconstant"

parameters = column_names.replace(" ", "_")
noblock = sys.argv[7]

base_model = os.path.basename(BASE_MODEL)
LORA_WEIGHTS = f"saved_model/{base_model}/{project}_{test_size}_{parameters}"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)


###Preparing Llama
max_length=2048
print(BASE_MODEL)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model, LORA_WEIGHTS, torch_dtype=torch.float16, force_download=True
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )
    
if device != "cpu":
    model.half()
model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)


def generate_prompt(instruction, input=None):
    
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

###
def evaluate(
    instruction,
    input=None,
    temperature=0,
    top_p=1,
    top_k=50,
    num_beams=2,
    max_new_tokens=max_length,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    test = tokenizer.encode(prompt)
    print(len(test))
    
    # decoded_input_ids = tokenizer.decode(input_ids)
    # length = len(decoded_input_ids)
    
    # print(type(decoded_input_ids))
    # print(decoded_input_ids)
    # print(length)
    
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        # top_k=top_k,
        # num_beams=num_beams,
        # repetition_penalty=1.1,
        # length_penalty=1,
        # repetition_penalty_sustain=256,
        # token_repetition_penalty_decay=128,
        # do_sample=True,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            
        )
    # with torch.no_grad():
    #     generation_output = model.generate(
    #         input_ids=input_ids,
    #         max_new_tokens=max_new_tokens,
    #         top_p=top_p,
    #         temperature=temperature,
    #         use_cache=True,
    #         top_k=top_k,
    #         repetition_penalty=1.0,
    #         length_penalty=1,
    #         return_dict_in_generate=True,
    #         # output_scores=True
    #     )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.strip()


###Start processing data
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

parameters_length = len(column_names.split())
column_1 = str(column_names.split()[0])
column_2 = str(column_names.split()[1])
if parameters_length == 3:
    column_3 = str(column_names.split()[2])

level_labels = [['trace'], ['debug'], ['info'], ['error'], ['warn']] #hardcoding all available from here to avoid error during n-1 comparison
redundant_columns = ['logcall', 'parameter', 'constant', 'callsite', 'line', 'type', 'name', 'pname', 'pparameter', 'pconstant'] #keeping level and label
if column_1 in redundant_columns: redundant_columns.remove(column_1)
if column_2 in redundant_columns: redundant_columns.remove(column_2)

if parameters_length == 3:
    redundant_columns.remove(column_3)
    

def generate_response(template, name):
    return template.format(name)

arguments = "_".join(["finetuned", project, split_type, file_index, base_model, str(test_size), parameters, noblock.replace("_", "")])

output_file = os.path.join(main_path, "results", "output_" + arguments + ".json")

if os.path.exists(output_file):
    os.remove(output_file)
    
with open(output_file, 'w') as f:    
    f.write("")
# #     print("Project: " + project, file=f)
# #     print("num_epochs: " + str(num_epochs), file=f)
# #     print("model_name: " + model_name, file=f)
# #     print("BASE_MODEL: " + os.path.basename(BASE_MODEL), file=f)
# #     print("test_size: " + str(test_size), file=f)
# #     print("input: " + str(column_names), file=f)

project_start_time = time.time()


df = pd.read_json(os.path.join(main_path, "merged", "merged_" + split_type + "_" + str(test_size) + "_" + project + "_" + file_index + ".json"))

###Commented away because running with this will cause memory problem
###We cannot read the whole file in one go
# df = pd.read_csv(os.path.join(main_path, "merged_raw", "merged_" + split_type + "_" + str(test_size) + "_" + project + ".csv"))

df_results = pd.DataFrame(columns=["project", "file_index", "run_type", "test_size", "index", "prediction"])

for row_index, row in tqdm(df.iterrows(), total=len(df)):
    
    template = ""
    instruction = ""
    prediction = "NA"
    if parameters_length == 3:
        row1 = row[column_1]
        row2 = row[column_2]
        row3 = row[column_3]
        
        instruction = "Choose a suitable log level for the code with syntactic and semantic information provided."
        template = f"The code syntactic is {row1}, the code semantic is {row2}, and the log message is {row3}."
        # template = "Given the source code is " + row[column_1] + " and the log message is " + row[column_2] + ". Between debug, warn, error, trace, info, and fatal, which is the appropriate log level for this logging statement?" 
        
    elif parameters_length == 2:
        row1 = row[column_1]
        row2 = row[column_2]
        
        instruction = "Choose a suitable log level for the source code provided."
        template = f"The source code is {row1}, and the log message is {row2}." 
        
        # template = "Given the syntactic is " + row[column_1] + ", the semantic is " + row[column_2] + " and the log message is " + row[column_3] + ". Between debug, warn, error, trace, info, and fatal, which is the appropriate log level for this logging statement?"
    
    prediction = evaluate(instruction=instruction, input=template)
    index = prediction.find("The log level is ")
    
    if index != -1:
        # Extract the part of the string after "The log level is "
        prediction = prediction[index + len("The log level is "):]
    
        # Remove " .</s>" from the extracted string
        prediction = prediction.replace(".</s>", "")
    
    
        row = [project, file_index, split_type, test_size, row["index"], prediction]
        df_results.loc[len(df.index)] = row
        
    df_results.to_json(output_file, orient='records')
        
    # df = df.drop(columns=[col for col in df if col not in ['template']])
    
    # # Apply the function to create the 'response' column
    # # df['response'] = df['template'].apply(lambda x: evaluate(input=x))
    
    # df['response'].to_json(output_file, orient='records')
