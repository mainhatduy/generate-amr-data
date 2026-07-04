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

---

## Running the Diverse Sampling Pipeline

The [diverse_sampling_pipeline.py](file:///teamspace/studios/this_studio/generate-amr-data/diverse_sampling_pipeline.py) script is designed to generate top-k diverse reasoning paths for AMR parsing.

### How it works:
1. **Stage 1 (Generation)**: Generates $n$ raw reasoning paths per sentence using vLLM's batched n-sampling and writes them to an intermediate raw samples file (`data/diverse_reasoning_results.raw_samples.jsonl`).
2. **Stage 2 (Embedding & Selection)**:
   - Evaluates all generated reasoning paths against gold AMR using `smatchpp` to calculate F1 scores.
   - Filters out samples with F1 < threshold (default `85.0`).
   - Batches the embedding generation for all valid reasoning paths using a vLLM pooling model (`runner="pooling"`) for high performance.
   - Selects the top-k diverse reasoning paths via MMR (Maximal Marginal Relevance).
   - Writes the selected candidates to the final output file (`data/diverse_reasoning_results.jsonl`), excluding the heavy raw `full_response` to keep the file size extremely small.

### Skip & Resume Logic on Reruns:
- **Completed records**: If a sentence has already achieved at least $k$ reasoning paths with F1 >= threshold in the final output file, the pipeline skips it entirely (saving GPU/API compute).
- **Partially generated records**: If a record has generated raw responses in the raw samples file but hasn't completed Stage 2 selection, Stage 1 skips generation and Stage 2 performs the scoring and diversity selection.

### Execution:

#### 1. Run the entire pipeline sequentially (Stage 1 + Stage 2):
```bash
uv run python diverse_sampling_pipeline.py \
  --config configs/diverse_sampling_config.json \
  --output data/diverse_reasoning_results.jsonl
```

#### 2. Run Stage 1 (Generation) only:
```bash
uv run python diverse_sampling_pipeline.py \
  --config configs/diverse_sampling_config.json \
  --output data/diverse_reasoning_results.jsonl \
  --stage 1
```

#### 3. Run Stage 2 (Embedding & Selection) only:
```bash
uv run python diverse_sampling_pipeline.py \
  --config configs/diverse_sampling_config.json \
  --output data/diverse_reasoning_results.jsonl \
  --stage 2
```

#### 4. Override parameters via CLI:
```bash
uv run python diverse_sampling_pipeline.py \
  --config configs/diverse_sampling_config.json \
  --output data/diverse_reasoning_results.jsonl \
  --f1-threshold 85.0 \
  --max-samples 100
```
