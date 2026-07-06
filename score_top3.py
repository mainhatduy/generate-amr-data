import argparse
import json
import os
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

from utils.amr_utils import extract_amr, extract_thinking, fix_amr_parentheses

# Suppress warnings and logger logs from Penman and Smatchpp in the main process
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("smatchpp").setLevel(logging.ERROR)
logging.getLogger("penman").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Parallel Worker
# ---------------------------------------------------------------------------

_worker_scorer = None

def _init_worker_scorer():
    """Initializer for workers — creates a per-process Smatchpp."""
    global _worker_scorer
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.WARNING)
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("__main__").setLevel(logging.ERROR)
    logging.getLogger("smatchpp").setLevel(logging.ERROR)
    logging.getLogger("penman").setLevel(logging.ERROR)
    from smatchpp import Smatchpp
    _worker_scorer = Smatchpp()


def _score_record_worker(args: Tuple[int, str, str, List[str], float]) -> Dict[str, Any]:
    global _worker_scorer
    idx, sentence, gold_amr, responses, f1_threshold = args

    scored_items = []
    if responses is not None:
        for resp in responses:
            if not resp or not resp.strip():
                continue
            thinking = extract_thinking(resp)
            pred_amr = extract_amr(resp)
            if not pred_amr:
                continue
            pred_amr = fix_amr_parentheses(pred_amr)

            try:
                result = _worker_scorer.score_pair(gold_amr, pred_amr)
                scores = result["main"]
                f1 = float(scores.get("F1", 0.0))
                precision = float(scores.get("Precision", 0.0))
                recall = float(scores.get("Recall", 0.0))
                
                scored_items.append({
                    "thinking": thinking,
                    "pred_amr": pred_amr,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall
                })
            except Exception:
                continue

    # Sort by F1 in descending order
    scored_items.sort(key=lambda x: x["f1"], reverse=True)
    
    # Pick the top 3 highest F1 samples
    top_3 = scored_items[:3]
    best_f1 = top_3[0]["f1"] if top_3 else 0.0
    
    # Check if complete (at least 3 samples with F1 >= f1_threshold)
    valid_samples = [s for s in top_3 if s["f1"] >= f1_threshold]
    is_complete = len(valid_samples) >= 3

    return {
        "id": idx,
        "sentence": sentence,
        "gold_amr": gold_amr,
        "selected_samples": top_3,
        "total_generated": len(responses) if responses else 0,
        "best_f1": best_f1,
        "is_complete": is_complete
    }

# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

def load_and_merge_raw_samples(raw_path: Path) -> List[Dict[str, Any]]:
    """
    Load raw responses and merge duplicate indices/sentences to accumulate responses.
    """
    merged_data = {}
    if not raw_path.exists():
        raise FileNotFoundError(f"Input file not found: {raw_path}")
        
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                key = data["index"] if "index" in data else data["sentence"]
                responses = data.get("responses", [])
                
                if key in merged_data:
                    # Append new responses, avoiding duplicates
                    seen = set(merged_data[key]["responses"])
                    for r in responses:
                        if r not in seen:
                            merged_data[key]["responses"].append(r)
                            seen.add(r)
                else:
                    merged_data[key] = {
                        "index": data.get("index"),
                        "sentence": data["sentence"],
                        "gold_amr": data["gold_amr"],
                        "responses": list(responses)
                    }
            except Exception:
                continue
                
    return list(merged_data.values())


def main():
    parser = argparse.ArgumentParser(
        description="Score raw samples and select top-3 highest F1 reasoning paths."
    )
    parser.add_argument(
        "--input",
        default="data/diverse_reasoning_results.raw_samples.jsonl",
        help="Path to input raw samples JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/top3_reasoning_results.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=85.0,
        help="F1 threshold for checking completeness (default: 85.0)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    f1_threshold = args.f1_threshold

    print(f"Loading and merging raw samples from {input_path}...")
    records = load_and_merge_raw_samples(input_path)
    print(f"Total unique records to score: {len(records)}")

    # Build work items: (index, sentence, gold_amr, responses, f1_threshold)
    work_items = [
        (r["index"], r["sentence"], r["gold_amr"], r["responses"], f1_threshold)
        for r in records
    ]

    num_workers = min(os.cpu_count() or 4, len(records))
    print(f"Scoring in parallel with {num_workers} workers...")
    
    results = []
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
            desc="Scoring (smatch)",
            unit="rec",
        ):
            results.append(result)

    # Sort output by index (if available) or sentence
    results.sort(key=lambda x: x["id"] if x["id"] is not None else x["sentence"])

    # Save to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for res in results:
            writer.write(json.dumps(res, ensure_ascii=False) + "\n")

    # Compute final statistics
    total_records = len(results)
    total_complete = sum(1 for r in results if r["is_complete"])
    avg_best_f1 = sum(r["best_f1"] for r in results) / total_records if total_records > 0 else 0.0

    print("\n" + "=" * 50)
    print(f"  Total records  : {total_records}")
    print(f"  Complete (>=3) : {total_complete} / {total_records}")
    print(f"  Avg best F1    : {avg_best_f1:.2f}")
    print(f"  Saved to       : {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
