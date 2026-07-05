from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import os
import re
import sys
import json
from datetime import datetime
import torch
import wandb
import penman
from datasets import load_dataset
from dotenv import load_dotenv
from trl import SFTTrainer, SFTConfig

# Load configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "../configs/hyperparameters.json")
with open(config_path, "r") as f:
    config = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["environment"]["cuda_visible_devices"]

load_dotenv()

# GPU check
print(f"Có GPU không? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch không tìm thấy GPU.")

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config["model"]["model_name"],
    max_seq_length = config["model"]["max_seq_length"],
    full_finetuning = config["model"]["full_finetuning"],
    token = os.getenv("HF_TOKEN")
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = config["dataset"]["chat_template"],
)

# Helper functions for AMR processing
def remove_wiki(amr_text):
    triples = penman.decode(amr_text).triples
    triples = [x for x in triples if x[1] != ":wiki"]
    return penman.encode(penman.Graph(triples))

def flat_amr(amr_text):
    graph = penman.decode(amr_text)
    return penman.encode(graph, indent=0, compact=True).replace("\n", " ")

# Helper for prompt building
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
from services.amr_hint.prompt_builder import build_prompt

def clean_model_respose(response):
    response = re.sub(r"<amr>.*?</amr>", "", response, flags=re.DOTALL)
    response = response.replace("<think>", "").replace("</think>", "")
    return response.strip()

def generate_conversation(examples):
    inputs  = examples["sentence"]
    outputs = examples["gold_amr"]
    selected_samples_list = examples["selected_samples"]

    conversations = []
    for input_text, output, selected_samples in zip(inputs, outputs, selected_samples_list):
        system_prompt = build_prompt(input_text, output)
        cleaned_output = remove_wiki(output)
        cleaned_output = flat_amr(cleaned_output)

        user_msg = (
            "Convert the following English sentence into its Abstract Meaning"
            f" Representation (AMR):\n\n<sentence>{input_text}</sentence>"
        )
        for sample in selected_samples:
            raw_reasoning = sample["thinking"]
            reasoning = clean_model_respose(raw_reasoning)
            assistant_msg = f"<think>{reasoning.strip()}</think>\n\n<amr>{cleaned_output}</amr>"
            conversations.append([
                # {"role" : "system",     "content" : system_prompt},
                {"role" : "user",      "content" : user_msg},
                {"role" : "assistant", "content" : assistant_msg},
            ])

    return { "conversations": conversations, }

# Deduplicate input sentences
seen_texts = set()
def filter_duplicates(example):
    if example["sentence"] in seen_texts:
        return False
    seen_texts.add(example["sentence"])
    return True

# Load dataset
data_path = os.path.join(script_dir, "../data/top3_reasoning_results.jsonl")
dataset = load_dataset("json", data_files=data_path).shuffle(seed=42)

dataset = dataset.filter(filter_duplicates)
dataset = dataset.map(generate_conversation, batched = True, remove_columns=dataset["train"].column_names, num_proc = 32)

# Apply chat template
def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True, num_proc = 32)

# Wandb initialization
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
tempt = config['model']['model_name'].split("/")[-1].lower()
run_name = f"{tempt}-amr-{timestamp}"
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key)

run = wandb.init(
    project="sft-llm-amr-parsing",
    name=run_name
)

# Trainer setup
training_config = config["training"]

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = training_config["dataset_text_field"],
        packing = training_config["packing"],
        per_device_train_batch_size = training_config["per_device_train_batch_size"],
        gradient_accumulation_steps = training_config["gradient_accumulation_steps"],
        warmup_ratio = training_config["warmup_ratio"],
        num_train_epochs = training_config["num_train_epochs"],
        learning_rate = training_config["learning_rate"],
        logging_steps = training_config["logging_steps"],
        optim = training_config["optim"],
        weight_decay = training_config["weight_decay"],
        lr_scheduler_type = training_config["lr_scheduler_type"],
        report_to = training_config["report_to"],
    ),
)

# Train on responses only
response_config = config["train_on_responses"]
trainer = train_on_responses_only(
    trainer,
    instruction_part = response_config["instruction_part"],
    response_part = response_config["response_part"],
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Run training
trainer_stats = trainer.train()

# Push full model to HF Hub
model_id = f"viamr-project/{run_name}"
model.push_to_hub(model_id, token = os.getenv("HF_TOKEN"))
tokenizer.push_to_hub(model_id, token = os.getenv("HF_TOKEN"))

print(f"✓ Full model đã được push lên: {run_name}")
print("Model này có thể dùng trực tiếp với vLLM!")
