"""
diverse_sampling_pipeline.py — Pipeline tìm top-k diverse reasoning paths cho AMR parsing.

Workflow:
    1. Load model qua vLLM
    2. Cho mỗi input sentence, generate n samples (n-sampling)
    3. Chấm Smatch F1 bằng smatchpp
    4. Embed thinking processes → chọn top-k diverse bằng MMR
    5. Lưu kết quả, hỗ trợ skip khi rerun

Cách sử dụng:
    uv run python diverse_sampling_pipeline.py \\
        --config configs/diverse_sampling_config.json \\
        --output data/diverse_reasoning_results.jsonl \\
        [--max-samples N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from smatchpp import Smatchpp

from schema.data_schema import DiverseReasoningResult, ReasoningSample
from services.vllm.engine import VLLMEngine


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("configs/diverse_sampling_config.json")
DEFAULT_OUTPUT_PATH = Path("data/diverse_reasoning_results.jsonl")

USER_PROMPT = (
    "Convert the following English sentence into its Abstract Meaning "
    "Representation (AMR):\n\n"
    "<sentence>{sentence}</sentence>"
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def extract_thinking(response: str) -> str:
    """Extract thinking process before </think> tag."""
    if "</think>" in response:
        thinking = response.split("</think>", 1)[0]
        # Remove the <think> tag if it exists at the start or inside
        thinking = thinking.replace("<think>", "")
        return thinking.strip()
    
    # Fallback in case </think> is missing but <think> is present
    if "<think>" in response:
        thinking = response.split("<think>", 1)[1]
        return thinking.strip()
        
    # If no tags are present, assume thinking is everything except the AMR block
    thinking = re.sub(r"<amr>.*?</amr>", "", response, flags=re.DOTALL)
    return thinking.strip()


def extract_amr(response: str) -> str | None:
    """Extract content between <amr>...</amr> tags."""
    match = re.search(r"<amr>(.*?)</amr>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback to capture anything after <amr> if </amr> is missing due to truncation
    if "<amr>" in response:
        return response.split("<amr>", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_amr_pair(gold: str, pred: str, scorer: Smatchpp) -> Dict[str, float]:
    """Score a single (gold, pred) AMR pair with smatchpp."""
    try:
        result = scorer.score_pair(gold, pred)
        return result["main"]  # {'F1': ..., 'Precision': ..., 'Recall': ...}
    except Exception as e:
        return {"F1": 0.0, "Precision": 0.0, "Recall": 0.0, "error": str(e)}


# ---------------------------------------------------------------------------
# MMR diversity selection
# ---------------------------------------------------------------------------


def select_diverse_mmr(
    samples: List[ReasoningSample],
    embeddings: np.ndarray,
    top_k: int = 3,
    diversity_weight: float = 0.5,
) -> List[ReasoningSample]:
    """
    Select top-k diverse samples using Maximal Marginal Relevance (MMR).

    The score for each candidate is:
        score = λ * normalized_f1 - (1-λ) * max_similarity_to_selected

    Args:
        samples: List of scored ReasoningSample objects.
        embeddings: (N, D) numpy array of thinking embeddings.
        top_k: Number of diverse samples to select.
        diversity_weight: λ ∈ [0, 1]. Higher = more weight on F1.

    Returns:
        List of top_k selected ReasoningSample objects.
    """
    if len(samples) <= top_k:
        return sorted(samples, key=lambda s: s.f1, reverse=True)

    n = len(samples)
    lam = diversity_weight

    # Normalize F1 to [0, 1]
    f1_scores = np.array([s.f1 for s in samples])
    f1_max = f1_scores.max()
    f1_min = f1_scores.min()
    if f1_max > f1_min:
        f1_norm = (f1_scores - f1_min) / (f1_max - f1_min)
    else:
        f1_norm = np.ones(n)

    # Cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)  # avoid division by zero
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    selected_indices: List[int] = []
    remaining = set(range(n))

    # Step 1: Pick the sample with highest F1 as seed
    seed_idx = int(np.argmax(f1_scores))
    selected_indices.append(seed_idx)
    remaining.discard(seed_idx)

    # Step 2: Greedily pick K-1 more
    for _ in range(top_k - 1):
        if not remaining:
            break

        best_score = -float("inf")
        best_idx = -1

        for cand_idx in remaining:
            # Max similarity to any already-selected sample
            max_sim = max(sim_matrix[cand_idx, sel] for sel in selected_indices)
            score = lam * f1_norm[cand_idx] - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_idx = cand_idx

        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining.discard(best_idx)

    return [samples[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _model_to_json_line(model: DiverseReasoningResult) -> str:
    """Serialize a Pydantic model to a JSON line."""
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(ensure_ascii=False)
    return model.json(ensure_ascii=False)


def load_completed_results(
    output_path: Path,
) -> Dict[str, DiverseReasoningResult]:
    """
    Load already-processed results from the output JSONL file.

    Returns a dict mapping sentence → DiverseReasoningResult.
    """
    results: Dict[str, DiverseReasoningResult] = {}
    if not output_path.exists():
        return results
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                result = DiverseReasoningResult(**data)
                results[result.sentence] = result
            except (json.JSONDecodeError, Exception):
                continue
    return results


def _iter_batches(
    dataset: Iterable[Dict[str, Any]],
    batch_size: int,
    max_samples: int | None,
) -> Iterable[List[Dict[str, Any]]]:
    """Yield successive batches from dataset."""
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


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_diverse_sampling_pipeline(
    config_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> Dict[str, int]:
    """
    Main pipeline:
        1. Load model → load dataset
        2. Skip already-complete samples
        3. Generate n samples per input via vLLM n-sampling
        4. Score each with smatchpp
        5. Embed thinking processes
        6. Select top-k diverse via MMR
        7. Save to JSONL (rewrite mode for correctness)

    Returns processing statistics.
    """
    load_dotenv()
    config = load_config(config_path)

    # --- Env ---
    env_config = config.get("env", {})
    cuda_device = env_config.get("CUDA_VISIBLE_DEVICES", "0")

    # --- Pipeline config ---
    pipeline_config = config.get("pipeline_config", {})
    n_samples = int(pipeline_config.get("n_samples_per_input", 16))
    top_k = int(pipeline_config.get("top_k_diverse", 3))
    f1_threshold = float(pipeline_config.get("f1_threshold", 0.0))
    batch_size = int(pipeline_config.get("batch_size", 64))
    skip_complete = bool(pipeline_config.get("skip_complete", True))

    # --- Embedding config ---
    embedding_config = config.get("embedding_config", {})
    embed_model_name = embedding_config.get("model", "all-MiniLM-L6-v2")
    diversity_weight = float(embedding_config.get("diversity_weight", 0.5))

    # --- Load vLLM engine ---
    print("[1/5] Starting vLLM engine...")
    model_config = config.get("model_config", {})
    sampling_config = config.get("sampling_config", {})
    engine_config = {**model_config, **sampling_config}
    engine = VLLMEngine(config=engine_config, cuda_device=cuda_device)

    # --- Load embedding model ---
    print("[2/5] Loading sentence embedding model...")
    embed_model = SentenceTransformer(embed_model_name)
    print(f"  ✅ Embedding model loaded: {embed_model_name}")

    # --- Load dataset ---
    print("[3/5] Loading dataset from Hugging Face...")
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

    # --- Load existing results for skip/resume ---
    print("[4/5] Checking for existing results...")
    existing_results = load_completed_results(output_path)
    complete_count = sum(1 for r in existing_results.values() if r.is_complete)
    if existing_results:
        print(f"  ↳ Found {len(existing_results)} existing results ({complete_count} complete)")

    # --- Generate & select ---
    print(f"[5/5] Generating {n_samples} sample(s) per input, selecting top-{top_k} diverse...")
    scorer = Smatchpp()

    stats = {"processed": 0, "skipped": 0, "success": 0, "failed": 0}
    all_results = dict(existing_results)  # start from existing

    for batch in _iter_batches(dataset, batch_size, max_samples):
        # --- Filter batch: skip complete, collect valid ---
        valid_samples: List[Dict[str, Any]] = []
        prompts: List[List[Dict[str, str]]] = []

        for sample in batch:
            sentence = sample.get(sentence_field)
            amr = sample.get(amr_field)

            if not isinstance(sentence, str) or not isinstance(amr, str):
                stats["failed"] += 1
                continue

            # Skip if already complete
            if skip_complete and sentence in all_results:
                existing = all_results[sentence]
                if existing.is_complete:
                    stats["skipped"] += 1
                    continue

            user_prompt = USER_PROMPT.format(sentence=sentence)
            messages = [{"role": "user", "content": user_prompt}]

            valid_samples.append(sample)
            prompts.append(messages)

        if not prompts:
            continue

        # --- Generate n samples per prompt ---
        try:
            batch_responses: List[List[str]] = engine.generate_batch_n_samples(
                prompts, n=n_samples
            )
        except Exception as exc:
            print(f"  ❌ Batch generation failed: {exc}")
            stats["failed"] += len(prompts)
            continue

        # --- Process each input ---
        for sample, responses in zip(valid_samples, batch_responses):
            sentence = sample[sentence_field]
            gold_amr = sample[amr_field].strip()
            stats["processed"] += 1

            # Score each response
            scored_samples: List[ReasoningSample] = []
            for resp in responses:
                if not resp or not resp.strip():
                    continue

                thinking = extract_thinking(resp)
                pred_amr = extract_amr(resp)

                if not pred_amr:
                    continue

                scores = score_amr_pair(gold_amr, pred_amr, scorer)
                f1 = float(scores.get("F1", 0.0))

                # Apply F1 threshold filter
                if f1 < f1_threshold:
                    continue

                scored_samples.append(
                    ReasoningSample(
                        thinking=thinking,
                        pred_amr=pred_amr,
                        full_response=resp,
                        f1=f1,
                        precision=float(scores.get("Precision", 0.0)),
                        recall=float(scores.get("Recall", 0.0)),
                    )
                )

            if not scored_samples:
                stats["failed"] += 1
                # Still save a record with empty selected_samples
                result = DiverseReasoningResult(
                    sentence=sentence,
                    gold_amr=gold_amr,
                    selected_samples=[],
                    total_generated=len(responses),
                    best_f1=0.0,
                    is_complete=False,
                )
                all_results[sentence] = result
                continue

            # --- Embed thinking processes ---
            thinking_texts = [s.thinking if s.thinking else s.pred_amr for s in scored_samples]
            embeddings = embed_model.encode(thinking_texts, convert_to_numpy=True)

            # --- Select top-k diverse via MMR ---
            selected = select_diverse_mmr(
                scored_samples, embeddings, top_k=top_k, diversity_weight=diversity_weight
            )

            best_f1 = max(s.f1 for s in scored_samples)
            is_complete = len(selected) >= top_k

            result = DiverseReasoningResult(
                sentence=sentence,
                gold_amr=gold_amr,
                selected_samples=selected,
                total_generated=len(responses),
                best_f1=best_f1,
                is_complete=is_complete,
            )
            all_results[sentence] = result
            stats["success"] += 1

            if stats["processed"] % 50 == 0:
                print(
                    f"  [{stats['processed']:>4}] "
                    f"success={stats['success']} | "
                    f"failed={stats['failed']} | "
                    f"skipped={stats['skipped']} | "
                    f"best_f1={best_f1:.1f} | "
                    f"{sentence[:60]}"
                )

    # --- Write all results (atomic rewrite) ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for result in all_results.values():
            writer.write(_model_to_json_line(result) + "\n")

    # --- Summary ---
    total_complete = sum(1 for r in all_results.values() if r.is_complete)
    total_results = len(all_results)
    avg_f1 = (
        sum(r.best_f1 for r in all_results.values()) / total_results
        if total_results > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print(f"  Total results  : {total_results}")
    print(f"  Complete       : {total_complete} / {total_results}")
    print(f"  Avg best F1    : {avg_f1:.2f}")
    print(f"  Processed      : {stats['processed']}")
    print(f"  Success        : {stats['success']}")
    print(f"  Failed         : {stats['failed']}")
    print(f"  Skipped (rerun): {stats['skipped']}")
    print(f"  Output         : {output_path}")
    print("=" * 70)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diverse reasoning paths for AMR parsing"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to output JSONL (default: {DEFAULT_OUTPUT_PATH})",
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
    run_diverse_sampling_pipeline(
        config_path=Path(args.config),
        output_path=Path(args.output),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
