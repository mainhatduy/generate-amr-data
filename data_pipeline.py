from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import ValidationError

from schema.data_schema import SystheticData
from services.amr_hint.prompt_builder import USER_PROMPT, build_prompt
from services.vllm import VLLMEngine


DEFAULT_CONFIG_PATH = Path("configs/configs.json")
DEFAULT_OUTPUT_PATH = Path("data/systhetic_data.jsonl")


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def _model_to_json_line(model: SystheticData) -> str:
    """Serialize a pydantic model to a single JSON line across v1/v2 APIs."""
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(ensure_ascii=False)
    return model.json(ensure_ascii=False)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_full_prompt(sentence: str, amr: str) -> str:
    """Combine system prompt (with hints) and sentence-specific user prompt."""
    system_prompt = build_prompt(sentence=sentence, amr=amr)
    user_prompt = USER_PROMPT.format(sentence=sentence)
    return f"{system_prompt}\n\n{user_prompt}"


def generate_systhetic_data(
    config_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> Dict[str, int]:
    load_dotenv()
    config = load_config(config_path)

    

    print("[1/4] Starting vLLM service...")
    vllm_config = config.get("vllm_config", {})
    engine = VLLMEngine(config=vllm_config)

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

    generation_config = config.get("generate_systhetic_data", {})
    responses_per_sample = int(generation_config.get("response_for_each_sample", 1))
    if responses_per_sample <= 0:
        raise ValueError("'response_for_each_sample' must be greater than 0.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[3/4] Generating model responses and saving successful samples...")
    total = 0
    success = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as writer:
        for sample in dataset:
            if max_samples is not None and total >= max_samples:
                break

            total += 1

            sentence = sample.get(sentence_field)
            amr = sample.get(amr_field)
            if not sentence or not amr:
                failed += 1
                continue

            try:
                prompt = build_full_prompt(sentence=sentence, amr=amr)
                responses = []
                for _ in range(responses_per_sample):
                    response = engine.generate(prompt).strip()
                    if response:
                        responses.append(response)

                if not responses:
                    failed += 1
                    continue

                record = SystheticData(
                    amr=amr,
                    sentence=sentence,
                    model_respose=responses,
                )
                writer.write(_model_to_json_line(record) + "\n")
                success += 1
            except (ValidationError, Exception):
                failed += 1

            if total % 10 == 0:
                print(f"Processed {total} samples | success={success} | failed={failed}")

    print("[4/4] Done")
    print(f"Output file: {output_path}")
    print(f"Processed: {total} | Success: {success} | Failed: {failed}")
    return {"processed": total, "success": success, "failed": failed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AMR synthetic data with vLLM")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_systhetic_data(
        config_path=Path(args.config),
        output_path=Path(args.output),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
