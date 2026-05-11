from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import ValidationError

from schema.data_schema import SystheticData
from services.amr_hint.prompt_builder import USER_PROMPT, build_prompt
from services.deepseek.deepseek import DeepSeekEngine


DEFAULT_CONFIG_PATH = Path("configs/configs.json")
DEFAULT_OUTPUT_PATH = Path("data/systhetic_data.jsonl")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _model_to_json_line(model: SystheticData) -> str:
    """Serialize a pydantic model to a single JSON line across v1/v2 APIs."""
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(ensure_ascii=False)
    return model.json(ensure_ascii=False)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_full_prompt(sentence: str, amr: str) -> list[dict]:
    """Build chat messages with system prompt (with hints) and sentence-specific user prompt."""
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


def _process_sample(
    sample: Dict[str, Any],
    sentence_field: str,
    amr_field: str,
    responses_per_sample: int,
    engine: DeepSeekEngine,
) -> Tuple[SystheticData | None, str | None]:
    sentence = sample.get(sentence_field)
    amr = sample.get(amr_field)
    if not sentence or not amr:
        return None, None

    try:
        prompt = build_full_prompt(sentence=sentence, amr=amr)
        responses: List[str] = []
        for _ in range(responses_per_sample):
            response = engine.generate(prompt).strip()
            if response:
                responses.append(response)

        if not responses:
            return None, None

        record = SystheticData(
            amr=amr,
            sentence=sentence,
            model_respose=responses,
        )
        return record, None
    except (ValidationError, Exception) as exc:
        return None, str(exc)


def generate_systhetic_data(
    config_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> Dict[str, int]:
    load_dotenv()
    config = load_config(config_path)

    

    print("[1/4] Starting DeepSeek service...")
    deepseek_config = config.get("deepseek_config", {})
    engine = DeepSeekEngine(config=deepseek_config)

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

    generation_config = config.get(
        "generate_systhetic_data",
        config.get("generate_synthetic_data", {}),
    )
    batch_size = int(generation_config.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("'batch_size' must be greater than 0.")
    responses_per_sample = int(generation_config.get("response_for_each_sample", 1))
    if responses_per_sample <= 0:
        raise ValueError("'response_for_each_sample' must be greater than 0.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_keys: set[tuple[str, str]] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as reader:
            for line in reader:
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
                    processed_keys.add((amr, sentence))

    print("[3/4] Generating model responses and saving successful samples...")
    total = 0
    success = 0
    failed = 0
    skipped = 0

    with output_path.open("a", encoding="utf-8") as writer:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for batch in _iter_batches(dataset, batch_size, max_samples):
                futures = {}
                for sample in batch:
                    sentence = sample.get(sentence_field)
                    amr = sample.get(amr_field)
                    if not isinstance(sentence, str) or not isinstance(amr, str):
                        failed += 1
                        continue
                    if (amr, sentence) in processed_keys:
                        skipped += 1
                        continue

                    total += 1
                    future = executor.submit(
                        _process_sample,
                        sample,
                        sentence_field,
                        amr_field,
                        responses_per_sample,
                        engine,
                    )
                    futures[future] = total

                for future in as_completed(futures):
                    sample_id = futures[future]
                    record, error = future.result()
                    if record is None:
                        failed += 1
                        if error:
                            print(f"Error on sample {sample_id}: {error}")
                        continue

                    writer.write(_model_to_json_line(record) + "\n")
                    processed_keys.add((record.amr, record.sentence))
                    success += 1

                if total % 10 == 0:
                    print(
                        "Processed "
                        f"{total} samples | success={success} | failed={failed} | skipped={skipped}"
                    )

    print("[4/4] Done")
    print(f"Output file: {output_path}")
    print(
        f"Processed: {total} | Success: {success} | Failed: {failed} | Skipped: {skipped}"
    )
    return {
        "processed": total,
        "success": success,
        "failed": failed,
        "skipped": skipped,
    }


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
