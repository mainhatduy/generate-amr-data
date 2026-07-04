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
import gc
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import torch

import numpy as np
from tqdm import tqdm
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


# -- Parallel scoring helpers ------------------------------------------------

_worker_scorer: Smatchpp | None = None


def _init_worker_scorer():
    """Initializer for ProcessPoolExecutor workers — creates a per-process Smatchpp."""
    global _worker_scorer
    # Suppress noisy smatchpp warnings (broken graph, :root relation, etc.)
    logging.getLogger("__main__").setLevel(logging.ERROR)
    logging.getLogger("smatchpp").setLevel(logging.ERROR)
    logging.getLogger("penman").setLevel(logging.ERROR)
    _worker_scorer = Smatchpp()


def _score_record_worker(
    args: Tuple[int, str, str, List[str], float],
) -> Tuple[int, str, str, int, List[Dict[str, Any]]]:
    """
    Score all responses for a single record in a worker process.

    Args:
        args: (idx, sentence, gold_amr, responses, f1_threshold)

    Returns:
        (idx, sentence, gold_amr, total_responses, scored_items)
        where scored_items is a list of dicts with keys:
            thinking, pred_amr, f1, precision, recall
        Only items with f1 >= f1_threshold are included.
    """
    global _worker_scorer
    idx, sentence, gold_amr, responses, f1_threshold = args

    scored_items: List[Dict[str, Any]] = []
    if responses is not None:
        for resp in responses:
            if not resp or not resp.strip():
                continue
            thinking = extract_thinking(resp)
            pred_amr = extract_amr(resp)
            if not pred_amr:
                continue

            try:
                result = _worker_scorer.score_pair(gold_amr, pred_amr)  # type: ignore[union-attr]
                scores = result["main"]
            except Exception:
                continue

            f1 = float(scores.get("F1", 0.0))
            if f1 < f1_threshold:
                continue

            scored_items.append(
                {
                    "thinking": thinking,
                    "pred_amr": pred_amr,
                    "f1": f1,
                    "precision": float(scores.get("Precision", 0.0)),
                    "recall": float(scores.get("Recall", 0.0)),
                }
            )

    return (idx, sentence, gold_amr, len(responses) if responses else 0, scored_items)


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
) -> Dict[Any, DiverseReasoningResult]:
    """
    Load already-processed results from the output JSONL file.

    Returns a dict mapping ID (int) or sentence (str) → DiverseReasoningResult.
    """
    results: Dict[Any, DiverseReasoningResult] = {}
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
                if result.id is not None:
                    results[result.id] = result
                else:
                    results[result.sentence] = result
            except (json.JSONDecodeError, Exception):
                continue
    return results


def load_raw_samples(raw_path: Path) -> Dict[Any, List[str]]:
    """
    Load raw responses from the intermediate raw samples file.
    Returns a dict mapping ID (int) or sentence (str) -> list of raw response strings.
    """
    raw_samples: Dict[Any, List[str]] = {}
    if not raw_path.exists():
        return raw_samples
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "index" in data:
                    raw_samples[data["index"]] = data["responses"]
                else:
                    raw_samples[data["sentence"]] = data["responses"]
            except Exception:
                continue
    return raw_samples


def save_raw_sample(raw_path: Path, index: int, sentence: str, gold_amr: str, responses: List[str]):
    """
    Append a raw generated sample to the intermediate file.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "index": index,
                    "sentence": sentence,
                    "gold_amr": gold_amr,
                    "responses": responses,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


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
    stage: str = "all",
    f1_threshold_override: float | None = None,
) -> Dict[str, int]:
    """
    Diverse sampling pipeline:
        Stage 1: Generate n samples for each record and save to intermediate raw_samples file.
        Stage 2: Score with smatchpp, embed reasoning paths (thinking) using vLLM,
                 run MMR diversity selection, and save the final results (removing full response).
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
    
    if f1_threshold_override is not None:
        f1_threshold = f1_threshold_override
    else:
        f1_threshold = float(pipeline_config.get("f1_threshold", 85.0))
        
    batch_size_val = pipeline_config.get("batch_size", 64)
    batch_size = int(batch_size_val) if batch_size_val is not None else None
    skip_complete = bool(pipeline_config.get("skip_complete", True))

    # --- Paths ---
    if output_path.suffix == ".jsonl":
        raw_path = output_path.with_name(output_path.stem + ".raw_samples.jsonl")
    else:
        raw_path = output_path.with_name(output_path.name + ".raw_samples.jsonl")

    # Load existing final results to check which sentences are already "complete"
    existing_results = load_completed_results(output_path)
    
    def is_sample_complete(idx: int, sent: str) -> bool:
        res = None
        if idx in existing_results:
            res = existing_results[idx]
        elif sent in existing_results:
            res = existing_results[sent]

        if res is None:
            return False
        if not res.is_complete:
            return False
        # A sentence is complete if it has at least top_k selected samples, all with f1 >= threshold
        valid_samples = [s for s in res.selected_samples if s.f1 >= f1_threshold]
        return len(valid_samples) >= top_k

    # --- Load dataset ---
    print("[1] Loading dataset...")
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

    stats = {"processed": 0, "skipped": 0, "success": 0, "failed": 0}
    all_results = dict(existing_results)

    # =========================================================================
    # STAGE 1: Generation
    # =========================================================================
    if stage in ("1", "all"):
        print("\n=== [Stage 1] Generating raw samples ===")
        raw_samples_dict = load_raw_samples(raw_path)
        
        sentences_to_generate = []
        samples_to_generate = []
        indices_to_generate = []
        
        inspected_count = 0
        for idx, sample in enumerate(dataset):
            if max_samples is not None and inspected_count >= max_samples:
                break
            inspected_count += 1
            
            sent = sample.get(sentence_field)
            amr = sample.get(amr_field)
            if not isinstance(sent, str) or not isinstance(amr, str):
                continue
                
            # Skip if complete in final output
            is_comp = is_sample_complete(idx, sent)
            if skip_complete and is_comp:
                stats["skipped"] += 1
                continue
                
            # Skip if already generated (with enough responses)
            # But if skip_complete is True and the sample is NOT complete, we WANT to resample it,
            # so we only skip if we are not resampling.
            if not (skip_complete and not is_comp):
                if idx in raw_samples_dict and len(raw_samples_dict[idx]) >= n_samples:
                    continue
                elif sent in raw_samples_dict and len(raw_samples_dict[sent]) >= n_samples:
                    continue
                
            sentences_to_generate.append(sent)
            samples_to_generate.append(sample)
            indices_to_generate.append(idx)

        print(f"Total dataset records: {len(dataset)}")
        print(f"Need generation: {len(sentences_to_generate)}")

        if len(sentences_to_generate) > 0:
            print("Starting vLLM generation engine...")
            model_config = config.get("model_config", {})
            sampling_config = config.get("sampling_config", {})
            engine_config = {**model_config, **sampling_config}
            engine = VLLMEngine(config=engine_config, cuda_device=cuda_device)

            # Process in batches
            actual_batch_size = batch_size if batch_size is not None else len(sentences_to_generate)
            if actual_batch_size <= 0:
                actual_batch_size = 1
            for i in range(0, len(sentences_to_generate), actual_batch_size):
                batch_samples = samples_to_generate[i : i + actual_batch_size]
                batch_indices = indices_to_generate[i : i + actual_batch_size]
                prompts = []
                for s in batch_samples:
                    user_prompt = USER_PROMPT.format(sentence=s[sentence_field])
                    prompts.append([{"role": "user", "content": user_prompt}])

                try:
                    batch_responses = engine.generate_batch_n_samples(prompts, n=n_samples)
                except Exception as exc:
                    print(f"  ❌ Batch generation failed: {exc}")
                    stats["failed"] += len(batch_samples)
                    continue

                for sample, index, responses in zip(batch_samples, batch_indices, batch_responses):
                    sent = sample[sentence_field]
                    gold_amr = sample[amr_field].strip()
                    save_raw_sample(raw_path, index, sent, gold_amr, responses)
                    stats["processed"] += 1

            # Shutdown vLLM generation engine to free GPU memory
            print("Shutting down vLLM generation engine to release memory...")
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            print("vLLM generation engine released.")
        else:
            print("All requested records are already generated.")

    # =========================================================================
    # STAGE 2: Embedding & Selection
    # =========================================================================
    if stage in ("2", "all"):
        print("\n=== [Stage 2] Scoring, Embedding & Selection ===")
        raw_samples_dict = load_raw_samples(raw_path)
        if not raw_samples_dict:
            print(f"No raw samples found in {raw_path}. Run Stage 1 first!")
            return stats

        # Filter out records that are already complete in final output (if skip_complete is True)
        records_to_process = []
        inspected_count = 0
        for idx, sample in enumerate(dataset):
            if max_samples is not None and inspected_count >= max_samples:
                break
            inspected_count += 1
            
            sent = sample.get(sentence_field)
            if not isinstance(sent, str):
                continue
            if skip_complete and is_sample_complete(idx, sent):
                continue
            if idx in raw_samples_dict or sent in raw_samples_dict:
                records_to_process.append((idx, sample))

        print(f"Need processing in Stage 2: {len(records_to_process)}")
        if not records_to_process:
            print("No records need processing in Stage 2.")
            return stats

        # Load embedding model configurations
        embedding_config = config.get("embedding_config", {})
        embed_model_name = embedding_config.get("model", "google/embeddinggemma-300m")
        diversity_weight = float(embedding_config.get("diversity_weight", 0.5))
        use_vllm_embed = bool(embedding_config.get("use_vllm", True))
        gpu_mem_util = float(embedding_config.get("gpu_memory_utilization", 0.9))

        print(f"Loading embedding model: {embed_model_name} (use_vllm={use_vllm_embed})")
        if use_vllm_embed:
            from vllm import LLM
            embed_model = LLM(
                model=embed_model_name,
                runner="pooling",
                gpu_memory_utilization=gpu_mem_util,
                enforce_eager=True,
                tensor_parallel_size=1,
                hf_token=os.getenv("HF_TOKEN"),
            )
        else:
            embed_model = SentenceTransformer(embed_model_name)

        # Step A: Parse and score F1 for all responses using parallel workers
        num_workers = min(os.cpu_count() or 4, len(records_to_process))
        print(f"Scoring generated samples with smatchpp ({num_workers} parallel workers)...")
        valid_samples_by_sentence = {}
        all_thinking_texts = []
        thinking_mapping = []

        # Build work items: (index, sentence, gold_amr, responses, f1_threshold)
        work_items = []
        for idx, record in records_to_process:
            sent = record[sentence_field]
            gold_amr = record[amr_field].strip()
            responses = raw_samples_dict.get(idx) or raw_samples_dict.get(sent)
            work_items.append((idx, sent, gold_amr, responses, f1_threshold))

        # Score in parallel with progress bar
        with ProcessPoolExecutor(
            max_workers=num_workers, initializer=_init_worker_scorer
        ) as executor:
            results_iter = executor.map(
                _score_record_worker, work_items,
                chunksize=max(1, len(work_items) // (num_workers * 4)),
            )
            for result in tqdm(
                results_iter,
                total=len(work_items),
                desc="  Scoring (smatch)",
                unit="rec",
                dynamic_ncols=True,
            ):
                idx, sent, gold_amr, total_responses, scored_items = result

                if not scored_items:
                    all_results[idx] = DiverseReasoningResult(
                        id=idx,
                        sentence=sent,
                        gold_amr=gold_amr,
                        selected_samples=[],
                        total_generated=total_responses,
                        best_f1=0.0,
                        is_complete=False,
                    )
                    continue

                scored_samples = [
                    ReasoningSample(
                        thinking=item["thinking"],
                        pred_amr=item["pred_amr"],
                        f1=item["f1"],
                        precision=item["precision"],
                        recall=item["recall"],
                    )
                    for item in scored_items
                ]

                valid_samples_by_sentence[idx] = scored_samples
                for sample_idx, s in enumerate(scored_samples):
                    text = s.thinking if s.thinking else s.pred_amr
                    # Truncate text to avoid exceeding vllm model context limit (e.g. 256 tokens)
                    words = text.split()
                    if len(words) > 100:
                        text = " ".join(words[:100])
                    all_thinking_texts.append(text)
                    thinking_mapping.append((idx, sample_idx))

        # Step B: Batch generate embeddings for all thinking processes
        if all_thinking_texts:
            print(f"Generating embeddings for {len(all_thinking_texts)} thinking processes in a single batch...")
            if use_vllm_embed:
                outputs = embed_model.embed(all_thinking_texts)
                embeddings = np.array([out.outputs.embedding for out in outputs])
            else:
                embeddings = embed_model.encode(all_thinking_texts, convert_to_numpy=True)

            # Map embeddings back to scored samples
            embeddings_by_sentence = {idx: [] for idx in valid_samples_by_sentence}
            for emb, (idx, sample_idx) in zip(embeddings, thinking_mapping):
                embeddings_by_sentence[idx].append(emb)
        else:
            embeddings_by_sentence = {}

        # Step C: Select top-k diverse reasoning paths for each sentence via MMR
        print("Selecting diverse reasoning paths via MMR...")
        for idx, scored_samples in valid_samples_by_sentence.items():
            sent_embeddings = np.array(embeddings_by_sentence[idx])
            selected = select_diverse_mmr(
                scored_samples, sent_embeddings, top_k=top_k, diversity_weight=diversity_weight
            )

            # Find matching record in records_to_process to get gold_amr and sentence
            matching_record = next(r for r in records_to_process if r[0] == idx)
            gold_amr = matching_record[1][amr_field].strip()
            sent = matching_record[1][sentence_field]

            best_f1 = max(s.f1 for s in scored_samples)
            is_complete = len(selected) >= top_k
            
            # Find total_generated
            total_generated = len(raw_samples_dict.get(idx) or raw_samples_dict.get(sent) or [])

            all_results[idx] = DiverseReasoningResult(
                id=idx,
                sentence=sent,
                gold_amr=gold_amr,
                selected_samples=selected,
                total_generated=total_generated,
                best_f1=best_f1,
                is_complete=is_complete,
            )
            stats["success"] += 1

        # Save final output in correct dataset order
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as writer:
            for idx, sample in enumerate(dataset):
                sent = sample.get(sentence_field)
                if idx in all_results:
                    writer.write(_model_to_json_line(all_results[idx]) + "\n")
                elif sent in all_results:
                    writer.write(_model_to_json_line(all_results[sent]) + "\n")

        # Cleanup embedding model memory
        if use_vllm_embed:
            print("Shutting down vLLM embedding engine...")
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass
            del embed_model
            gc.collect()
            torch.cuda.empty_cache()

    # --- Summary ---
    total_complete = sum(
        1 for r in all_results.values()
        if r.is_complete and len([s for s in r.selected_samples if s.f1 >= f1_threshold]) >= top_k
    )
    total_results = len(all_results)
    avg_f1 = (
        sum(r.best_f1 for r in all_results.values()) / total_results
        if total_results > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print(f"  Total results  : {total_results}")
    print(f"  Complete (F1>=85): {total_complete} / {total_results}")
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
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all"],
        default="all",
        help="Stage to run: 1 (generate raw), 2 (evaluate & MMR), all (both sequentially)",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=None,
        help="Override F1 threshold (e.g. 85.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diverse_sampling_pipeline(
        config_path=Path(args.config),
        output_path=Path(args.output),
        max_samples=args.max_samples,
        stage=args.stage,
        f1_threshold_override=args.f1_threshold,
    )


if __name__ == "__main__":
    main()
