import os
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import penman
from datetime import datetime
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config_path = "../configs/hyperparameters.json"
with open(config_path, "r") as f:
    config = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["environment"]["cuda_visible_devices"]

load_dotenv()


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

print(f"Có GPU không? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch không tìm thấy GPU.")


# ---------------------------------------------------------------------------
# Load model & tokenizer
# ---------------------------------------------------------------------------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model"]["model_name"],
    max_seq_length=config["model"]["max_seq_length"],
    full_finetuning=config["model"]["full_finetuning"],
    token=os.getenv("HF_TOKEN"),
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template=config["dataset"]["chat_template"],
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def remove_wiki(amr_text: str) -> str:
    """Strip :wiki roles from an AMR graph."""
    triples = penman.decode(amr_text).triples
    triples = [t for t in triples if t[1] != ":wiki"]
    return penman.encode(penman.Graph(triples))


def flat_amr(amr_text: str) -> str:
    """Flatten an AMR graph to a single line."""
    graph = penman.decode(amr_text)
    return penman.encode(graph, indent=0, compact=True).replace("\n", " ")


def generate_conversation(examples):
    inputs = examples["sentence"]
    reasonings = examples["reasoning"]
    outputs = examples["amr"]
    system_prompts = examples["system_prompt"]
    enable_thinking = config["inference"]["enable_thinking"]

    conversations = []
    for input_text, reasoning, output, system_prompt in zip(inputs, reasonings, outputs, system_prompts):
        output = remove_wiki(output)
        output = flat_amr(output)
        user_msg = (
            "Convert the following English sentence into its Abstract Meaning"
            f" Representation (AMR):\n\n<sentence>{input_text}</sentence>"
        )
        if enable_thinking:
            assistant_msg = f"<think>\n{reasoning}\n</think>\n\n<amr>\n{output}\n</amr>"
        else:
            assistant_msg = f"<think>\n\n</think>\n\n<amr>\n{output}\n</amr>"

        conversations.append([
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ])

    return {"conversations": conversations}


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}


def count_tokens(examples):
    token_counts = [
        len(tokenizer(text, truncation=False)["input_ids"])
        for text in examples["text"]
    ]
    return {"token_count": token_counts}


# ---------------------------------------------------------------------------
# Load & prepare dataset
# ---------------------------------------------------------------------------

dataset = load_dataset("json", data_files=config["dataset"]["train_file"]).shuffle(seed=42)
print(dataset)

# Test AMR helpers on one sample
sample = dataset["train"][1]["amr"]
print("Original:\n", sample)
print("Removed wiki:\n", remove_wiki(sample))
print("Flattened:\n", flat_amr(sample))

# Deduplicate by sentence
seen_texts: set = set()

def filter_duplicates(example):
    if example["sentence"] in seen_texts:
        return False
    seen_texts.add(example["sentence"])
    return True

dataset = dataset.filter(filter_duplicates)
dataset = dataset.map(generate_conversation, batched=True)
dataset = dataset.map(formatting_prompts_func, batched=True)

# Preview a formatted example
print(dataset["train"][100]["text"])


# ---------------------------------------------------------------------------
# Token distribution analysis
# ---------------------------------------------------------------------------

dataset_with_counts = dataset["train"].map(count_tokens, batched=True)
token_counts = dataset_with_counts["token_count"]

print(f"Tổng số mẫu: {len(token_counts)}")
print(f"Số token trung bình: {np.mean(token_counts):.2f}")
print(f"Số token trung vị: {np.median(token_counts):.2f}")
print(f"Số token min: {np.min(token_counts)}")
print(f"Số token max: {np.max(token_counts)}")
print(f"Độ lệch chuẩn: {np.std(token_counts):.2f}")
print("\nPercentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p}%: {np.percentile(token_counts, p):.0f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(token_counts, bins=50, edgecolor="black", alpha=0.7)
plt.xlabel("Số token")
plt.ylabel("Số lượng mẫu")
plt.title("Phân phối số token trong dataset")
plt.axvline(np.mean(token_counts), color="r", linestyle="--", label=f"Mean: {np.mean(token_counts):.0f}")
plt.axvline(np.median(token_counts), color="g", linestyle="--", label=f"Median: {np.median(token_counts):.0f}")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(token_counts, vert=True)
plt.ylabel("Số token")
plt.title("Box plot phân phối số token")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("token_distribution.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# W&B initialisation
# ---------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d-%H%M")
model_slug = config["model"]["model_name"].split("/")[-1].lower()
run_name = f"{model_slug}-amr-{timestamp}"

wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(project="sft-llm-amr-parsing", name=run_name)


# ---------------------------------------------------------------------------
# Trainer setup
# ---------------------------------------------------------------------------

training_config = config["training"]

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field=training_config["dataset_text_field"],
        packing=training_config["packing"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        warmup_ratio=training_config["warmup_ratio"],
        num_train_epochs=training_config["num_train_epochs"],
        learning_rate=training_config["learning_rate"],
        logging_steps=training_config["logging_steps"],
        optim=training_config["optim"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        report_to=training_config["report_to"],
    ),
)

# Mask instruction tokens so loss is computed on assistant responses only
response_config = config["train_on_responses"]
trainer = train_on_responses_only(
    trainer,
    instruction_part=response_config["instruction_part"],
    response_part=response_config["response_part"],
)

# Verify masking: decoded input and masked labels for row 100
print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
print(
    tokenizer.decode(
        [tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]
    ).replace(tokenizer.pad_token, " ")
)


# ---------------------------------------------------------------------------
# GPU memory snapshot before training
# ---------------------------------------------------------------------------

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

trainer_stats = trainer.train()


# ---------------------------------------------------------------------------
# Quick inference sanity-check
# ---------------------------------------------------------------------------

input_text = "The cat sits on the mat."
messages = [
    {
        "role": "user",
        "content": (
            "Convert the following English sentence into its Abstract Meaning"
            f" Representation (AMR):\n\n<sentence>{input_text}</sentence>"
        ),
    }
]

inference_config = config["inference"]
text = tokenizer.apply_chat_template(messages, tokenize=False)

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=inference_config["max_new_tokens"],
    temperature=inference_config["temperature"],
    top_p=inference_config["top_p"],
    top_k=inference_config["top_k"],
    streamer=TextStreamer(tokenizer, skip_prompt=False),
)


# ---------------------------------------------------------------------------
# Push model to Hugging Face Hub
# ---------------------------------------------------------------------------

model_id = f"viamr-project/{run_name}"
model.push_to_hub(model_id, token=os.getenv("HF_TOKEN"))
tokenizer.push_to_hub(model_id, token=os.getenv("HF_TOKEN"))

HfApi(token=os.getenv("HF_TOKEN")).upload_file(
    path_or_fileobj=config_path,
    path_in_repo=os.path.basename(config_path),
    repo_id=model_id,
    token=os.getenv("HF_TOKEN"),
)

print(f"✓ Full model đã được push lên: {run_name}")
print("Model này có thể dùng trực tiếp với vLLM!")


# ---------------------------------------------------------------------------
# Free VRAM
# ---------------------------------------------------------------------------

del trainer, model
gc.collect()
torch.cuda.empty_cache()

print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU Memory Reserved:  {torch.cuda.memory_reserved()  / 1024**3:.2f} GB")