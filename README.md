# Generate AMR Data

A pipeline to generate synthetic Abstract Meaning Representation (AMR) data using LLM services (DeepSeek, vLLM).

## Setup Environment

This project uses `uv` for Python package management. Follow the steps below to set up the virtual environment with Python 3.13.

### 1. Initialize and Sync Virtual Environment

Pin Python 3.13 and install all dependencies:

```bash
# Pin Python 3.13
uv python pin 3.13

# Create and synchronize the virtual environment
uv sync
```

Alternatively, you can initialize the sync specifying the Python version:
```bash
uv sync --python 3.13
```

### 2. Activate the Virtual Environment

Activate the newly created `.venv`:

```bash
source .venv/bin/activate
```

> [!TIP]
> If you are using vLLM and need `flash-attn` for accelerated performance, you can build and install it using:
> ```bash
> uv pip install ninja packaging
> MAX_JOBS=32 uv pip install flash-attn --no-build-isolation
> ```

---

## Configuration

### Set Environment Variables

Create a `.env` file in the root directory by copying the `.env.example` template:

```bash
cp .env.example .env
```

Open `.env` and configure your Hugging Face API token (required for downloading datasets):
```env
HF_TOKEN=your_huggingface_token_here
```

---

## Running the vLLM Pipeline

The [vllm_pipeline.py](file:///teamspace/studios/this_studio/generate-amr-data/vllm_pipeline.py) script executes the data generation pipeline using a local vLLM engine.

### 1. Run with Default Settings
By default, the script loads the configuration from [vllm_generation_config.json](file:///teamspace/studios/this_studio/generate-amr-data/configs/vllm_generation_config.json) and outputs generated data to `data/vllm_synthetic_data.jsonl`:

```bash
python vllm_pipeline.py
```

### 2. Run with Custom Parameters
You can specify custom configurations, a different output file, or cap the total number of dataset samples to process:

```bash
python vllm_pipeline.py \
  --config configs/vllm_generation_config.json \
  --output data/vllm_synthetic_data.jsonl \
  --max-samples 100
```
