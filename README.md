# Generate AMR Data

A pipeline to generate synthetic Abstract Meaning Representation (AMR) data using LLM services (DeepSeek, vLLM).

## Setup Environment

This project uses `uv` for Python package management.

### Dependency Groups

`unsloth` (fine-tuning) and `vllm` (inference) have **irreconcilable `transformers` version constraints** — they cannot coexist in the same environment:

| Constraint | `unsloth` / `unsloth-zoo` | `vllm >= 0.21` |
|---|---|---|
| `transformers` | `<=5.5.0` | `!=5.5.0` (needs `>=5.5.1`) |

To handle this, they are split into separate optional dependency groups:

| Group | Contents | Use case |
|-------|----------|----------|
| `train` | `unsloth`, `unsloth-zoo` | Fine-tuning jobs |
| `inference` | `vllm` | Inference / data generation |
| *(base)* | `matplotlib`, `penman`, `sentence-transformers`, etc. | Always installed |

### 1. Install for Inference (vLLM)

```bash
uv sync --group inference --link-mode copy
```

### 2. Install for Fine-tuning (Unsloth)

```bash
uv sync --group train --link-mode copy
```

### 3. Activate the Virtual Environment

```bash
source .venv/bin/activate
```

> [!IMPORTANT]
> If you encounter shared library compilation errors like `libcusparseLt.so.0: cannot open shared object file` or `libnvshmem_host.so.3: cannot open shared object file` when importing `torch` or `unsloth`, it means some cached `nvidia-*` packages were not linked correctly. Running with `--link-mode copy` (as shown above) fixes this by copying the dependencies cleanly instead of symlinking them.

> [!WARNING]
> Do **not** run `uv sync --group train --group inference` simultaneously — this will trigger the `transformers` version conflict. Install only the group you need for the current job.

> [!TIP]
> If you need `flash-attn` for accelerated performance (e.g., for Unsloth or vLLM), do **not** compile it from source to avoid CUDA version mismatches with the system compiler. Instead, install the prebuilt wheel matching Python 3.13 and PyTorch 2.10.0:
> ```bash
> uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
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
