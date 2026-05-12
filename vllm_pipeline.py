"""
vllm_pipeline.py — Pipeline sinh dữ liệu tổng hợp AMR sử dụng vLLM.

Cách sử dụng:
    python vllm_pipeline.py [--config PATH] [--output PATH] [--max-samples N]

Mặc định:
    --config  configs/vllm_generation_config.json
    --output  data/vllm_synthetic_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import ValidationError

from schema.data_schema import SystheticData
from services.amr_hint.prompt_builder import build_prompt
from services.vllm.engine import VLLMEngine


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("configs/vllm_generation_config.json")
DEFAULT_OUTPUT_PATH = Path("data/vllm_synthetic_data.jsonl")

# User prompt format — matches the SFT_Qwen3.ipynb notebook
USER_PROMPT = (
    "Convert the following English sentence into its Abstract Meaning Representation (AMR):\n\n"
    "<sentence>{sentence}</sentence>"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_to_json_line(model: SystheticData) -> str:
    """Serialize a Pydantic model to a single JSON line (v1 / v2 compatible)."""
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(ensure_ascii=False)
    return model.json(ensure_ascii=False)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_full_prompt(sentence: str, amr: str) -> List[Dict[str, str]]:
    """
    Build chat messages with a system prompt (containing AMR hints) and a
    sentence-specific user prompt.

    The format matches what was used in SFT_Qwen3.ipynb.
    """
    system_prompt = build_prompt(sentence=sentence, amr=amr)
    user_prompt = USER_PROMPT.format(sentence=sentence)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _iter_batches(
    dataset: Iterable[Dict[str, Any]],
    batch_size: int,
    max_samples: int | None,
) -> Iterable[List[Dict[str, Any]]]:
    """Yield successive batches from *dataset*, respecting *max_samples*."""
    batch: List[Dict[str, Any]] = []
    total = 0
    for sample in dataset:
        if max_samples is not None and total >= max_samples:
            break
        batch.append(sample)
        total += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_processed_keys(output_path: Path) -> set[Tuple[str, str]]:
    """Read already-processed (amr, sentence) pairs from an existing output file."""
    processed: set[Tuple[str, str]] = set()
    if not output_path.exists():
        return processed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            amr = record.get("amr")
            sentence = record.get("sentence")
            if isinstance(amr, str) and isinstance(sentence, str):
                processed.add((amr, sentence))
    return processed


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    config_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> Dict[str, int]:
    """
    Main pipeline: load model → load dataset → generate n samples per input
    → save to JSONL (with resume support).

    Returns a dict with processing statistics.
    """
    load_dotenv()
    config = load_config(config_path)

    # --- Apply environment overrides ---
    env_config = config.get("env", {})
    cuda_device = env_config.get("CUDA_VISIBLE_DEVICES", "0")

    # --- Load vLLM engine ---
    print("[1/4] Starting vLLM engine...")
    model_config = config.get("model_config", {})
    sampling_config = config.get("sampling_config", {})
    # Merge model + sampling params into one dict for VLLMEngine
    engine_config = {**model_config, **sampling_config}
    engine = VLLMEngine(config=engine_config, cuda_device=cuda_device)

    # --- Load dataset ---
    print("[2/4] Loading dataset from Hugging Face...")
    data_config = config.get("data_config", {})
    dataset_name = data_config.get("dataset")
    if not dataset_name:
        raise ValueError("Missing 'data_config.dataset' in config file.")

    split = data_config.get("split", "train")
    sentence_field = data_config.get("sentence_field", "sentence")
    amr_field = data_config.get("amr_field", "amr")

    dataset = load_dataset(
        dataset_name,
        split=split,
        token=os.getenv("HF_TOKEN"),
    )

    # --- Generation settings ---
    gen_config = config.get("generation_config", {})
    batch_size = int(gen_config.get("batch_size", 8))
    if batch_size <= 0:
        raise ValueError("'generation_config.batch_size' must be > 0.")
    n_samples = int(gen_config.get("n_samples_per_input", 1))
    if n_samples <= 0:
        raise ValueError("'generation_config.n_samples_per_input' must be > 0.")

    # --- Prepare output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_keys = _load_processed_keys(output_path)
    if processed_keys:
        print(f"  ↳ Resuming: {len(processed_keys)} samples already processed.")

    # --- Generate ---
    print(f"[3/4] Generating {n_samples} sample(s) per input (batch_size={batch_size})...")
    total = 0
    success = 0
    failed = 0
    skipped = 0

    with output_path.open("a", encoding="utf-8") as writer:
        for batch in _iter_batches(dataset, batch_size, max_samples):
            # Filter out already-processed and invalid samples
            valid_samples: List[Dict[str, Any]] = []
            prompts: List[List[Dict[str, str]]] = []

            for sample in batch:
                sentence = sample.get(sentence_field)
                amr = sample.get(amr_field)

                if not isinstance(sentence, str) or not isinstance(amr, str):
                    failed += 1
                    continue
                if (amr, sentence) in processed_keys:
                    skipped += 1
                    continue

                try:
                    prompt = build_full_prompt(sentence=sentence, amr=amr)
                except Exception as exc:
                    print(f"  ⚠ Prompt build failed for sentence '{sentence[:60]}': {exc}")
                    failed += 1
                    continue

                valid_samples.append(sample)
                prompts.append(prompt)
                total += 1

            if not prompts:
                continue

            # Generate n samples per prompt in a single batched vLLM call
            try:
                batch_responses: List[List[str]] = engine.generate_batch_n_samples(
                    prompts, n=n_samples
                )
            except Exception as exc:
                print(f"  ❌ Batch generation failed: {exc}")
                failed += len(prompts)
                continue

            for sample, responses in zip(valid_samples, batch_responses):
                sentence = sample[sentence_field]
                amr = sample[amr_field]

                responses = [r for r in responses if r and r.strip()]
                if not responses:
                    failed += 1
                    continue

                try:
                    record = SystheticData(
                        amr=amr,
                        sentence=sentence,
                        model_respose=responses,
                    )
                except ValidationError as exc:
                    print(f"  ❌ Validation error: {exc}")
                    failed += 1
                    continue

                writer.write(_model_to_json_line(record) + "\n")
                writer.flush()
                processed_keys.add((amr, sentence))
                success += 1

            if total % 50 == 0 and total > 0:
                print(
                    f"  Processed {total} | success={success} | "
                    f"failed={failed} | skipped={skipped}"
                )

    print("[4/4] Done.")
    print(f"  Output: {output_path}")
    print(
        f"  Processed: {total} | Success: {success} | "
        f"Failed: {failed} | Skipped: {skipped}"
    )
    return {
        "processed": total,
        "success": success,
        "failed": failed,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AMR synthetic data with vLLM"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to vLLM generation config JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to output JSONL file (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of dataset samples to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_synthetic_data(
        config_path=Path(args.config),
        output_path=Path(args.output),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
