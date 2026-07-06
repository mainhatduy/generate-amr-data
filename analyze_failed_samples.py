"""
analyze_failed_samples.py — Phân tích các sample trong raw_samples.jsonl
mà KHÔNG có predict AMR nào đạt F1 >= 85.

Workflow:
    1. Đọc diverse_reasoning_results.raw_samples.jsonl
    2. Chấm điểm tất cả predict AMR bằng smatchpp (parallel)
    3. Tìm các record mà max(F1) < 85
    4. Lưu vào data/failed_samples.jsonl để phân tích

Cách dùng:
    uv run python analyze_failed_samples.py \
        [--raw-samples data/diverse_reasoning_results.raw_samples.jsonl] \
        [--output data/failed_samples.jsonl] \
        [--f1-threshold 85.0] \
        [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils.amr_utils import extract_amr, fix_amr_parentheses

# ---------------------------------------------------------------------------
# Suppress noisy logs
# ---------------------------------------------------------------------------
logging.getLogger("smatchpp").setLevel(logging.ERROR)
logging.getLogger("penman").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Parallel scoring
# ---------------------------------------------------------------------------

_worker_scorer = None


def _init_worker():
    """Per-process Smatchpp initialiser."""
    global _worker_scorer
    from smatchpp import Smatchpp
    logging.getLogger("smatchpp").setLevel(logging.ERROR)
    logging.getLogger("penman").setLevel(logging.ERROR)
    _worker_scorer = Smatchpp()


def _score_record(
    args: Tuple[Any, str, str, List[str], float],
) -> Dict[str, Any]:
    """
    Score all responses for one record.
    Returns dict with scoring results.
    """
    global _worker_scorer
    idx, sentence, gold_amr, responses, f1_threshold = args

    scored_amrs: List[Dict[str, Any]] = []
    valid = 0

    for resp in (responses or []):
        if not resp or not resp.strip():
            continue
        pred_amr = extract_amr(resp)
        if not pred_amr:
            continue
        pred_amr = fix_amr_parentheses(pred_amr)
        valid += 1
        try:
            result = _worker_scorer.score_pair(gold_amr, pred_amr)  # type: ignore
            scores = result["main"]
            f1 = float(scores.get("F1", 0.0))
            precision = float(scores.get("Precision", 0.0))
            recall = float(scores.get("Recall", 0.0))
        except Exception:
            f1 = precision = recall = 0.0

        scored_amrs.append(
            {
                "pred_amr": pred_amr,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )

    max_f1 = max((s["f1"] for s in scored_amrs), default=0.0)

    return {
        "index": idx,
        "sentence": sentence,
        "gold_amr": gold_amr,
        "total_responses": len(responses) if responses else 0,
        "valid_responses": valid,
        "scored_amrs": scored_amrs,
        "max_f1": max_f1,
        "passes_threshold": max_f1 >= f1_threshold,
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_raw_samples(raw_path: Path) -> List[Dict[str, Any]]:
    """
    Load raw samples, merging duplicate indices.
    Returns list of dicts: {index, sentence, gold_amr, responses}.
    """
    merged: Dict[Any, Dict[str, Any]] = {}

    with raw_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading raw_samples.jsonl", unit="line"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = data.get("index") if "index" in data else data.get("sentence")
            if key is None:
                continue

            responses = data.get("responses", [])
            if key in merged:
                seen = set(merged[key]["responses"])
                for r in responses:
                    if r not in seen:
                        merged[key]["responses"].append(r)
                        seen.add(r)
            else:
                merged[key] = {
                    "index": data.get("index"),
                    "sentence": data.get("sentence", ""),
                    "gold_amr": data.get("gold_amr", ""),
                    "responses": list(responses),
                }

    return list(merged.values())


# ---------------------------------------------------------------------------
# Distribution helper
# ---------------------------------------------------------------------------


def _f1_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Bucket max_f1 scores into ranges."""
    buckets = {
        "0": 0, "0-10": 0, "10-20": 0, "20-30": 0, "30-40": 0,
        "40-50": 0, "50-60": 0, "60-70": 0, "70-80": 0, "80-85": 0,
        "85-90": 0, "90-95": 0, "95-100": 0,
    }
    for r in records:
        f = r["max_f1"]
        if f == 0:
            buckets["0"] += 1
        elif f < 10:
            buckets["0-10"] += 1
        elif f < 20:
            buckets["10-20"] += 1
        elif f < 30:
            buckets["20-30"] += 1
        elif f < 40:
            buckets["30-40"] += 1
        elif f < 50:
            buckets["40-50"] += 1
        elif f < 60:
            buckets["50-60"] += 1
        elif f < 70:
            buckets["60-70"] += 1
        elif f < 80:
            buckets["70-80"] += 1
        elif f < 85:
            buckets["80-85"] += 1
        elif f < 90:
            buckets["85-90"] += 1
        elif f < 95:
            buckets["90-95"] += 1
        else:
            buckets["95-100"] += 1
    return buckets


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze(
    raw_path: Path,
    output_path: Path,
    f1_threshold: float = 85.0,
    n_workers: Optional[int] = None,
) -> None:
    print(f"\n{'='*70}")
    print(f"  Analyzing raw samples from: {raw_path}")
    print(f"  F1 threshold            : {f1_threshold}")
    print(f"  Output (failed samples) : {output_path}")
    print(f"{'='*70}\n")

    # 1. Load
    records = load_raw_samples(raw_path)
    print(f"Total unique records loaded : {len(records)}")

    if not records:
        print("No records found. Exiting.")
        return

    # 2. Build work items
    work_items = [
        (
            rec["index"],
            rec["sentence"],
            rec["gold_amr"],
            rec["responses"],
            f1_threshold,
        )
        for rec in records
    ]

    # 3. Score in parallel
    workers = n_workers or min(os.cpu_count() or 4, len(records), 16)
    print(f"Scoring with smatchpp ({workers} workers)...")

    scored_results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_score_record, item): item for item in work_items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring", unit="rec"):
            try:
                scored_results.append(fut.result())
            except Exception as e:
                item = futures[fut]
                print(f"  Warning: Error scoring record {item[0]}: {e}")

    # Sort by index
    scored_results.sort(key=lambda x: (x["index"] is None, x["index"] if x["index"] is not None else ""))

    # 4. Separate pass / fail
    passed = [r for r in scored_results if r["passes_threshold"]]
    failed = [r for r in scored_results if not r["passes_threshold"]]

    total = len(scored_results)
    print(f"\n{'='*70}")
    print(f"  Total records scored       : {total}")
    print(f"  Passed (max_f1 >= {f1_threshold}): {len(passed)}")
    print(f"  Failed (max_f1 <  {f1_threshold}): {len(failed)}")
    if total > 0:
        print(f"  Failure rate               : {len(failed)/total*100:.2f}%")
        avg_max_f1 = sum(r["max_f1"] for r in scored_results) / total
        print(f"  Avg max_f1 (all)           : {avg_max_f1:.2f}")
        if failed:
            avg_fail_f1 = sum(r["max_f1"] for r in failed) / len(failed)
            print(f"  Avg max_f1 (failed only)   : {avg_fail_f1:.2f}")
    print(f"{'='*70}\n")

    # 5. Save failed samples (without full scored_amrs to keep file manageable)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in failed:
            rec_out = {
                "index": rec["index"],
                "sentence": rec["sentence"],
                "gold_amr": rec["gold_amr"],
                "total_responses": rec["total_responses"],
                "valid_responses": rec["valid_responses"],
                "max_f1": rec["max_f1"],
                "f1_threshold": f1_threshold,
                # Top 5 best AMRs for analysis
                "best_amrs": sorted(rec["scored_amrs"], key=lambda x: x["f1"], reverse=True)[:5],
            }
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    print(f"Failed samples saved to: {output_path}")
    print(f"  {len(failed)} records with no predict AMR achieving F1 >= {f1_threshold}")

    # 6. Save summary stats
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "raw_samples_path": str(raw_path),
        "f1_threshold": f1_threshold,
        "total_records": total,
        "passed": len(passed),
        "failed": len(failed),
        "failure_rate_pct": round(len(failed) / total * 100, 4) if total > 0 else 0,
        "avg_max_f1_all": round(sum(r["max_f1"] for r in scored_results) / total, 4) if total > 0 else 0,
        "avg_max_f1_failed": (
            round(sum(r["max_f1"] for r in failed) / len(failed), 4) if failed else None
        ),
        "f1_distribution": _f1_distribution(scored_results),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary stats saved to: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score all predict AMRs in raw_samples.jsonl and find failed samples"
    )
    parser.add_argument(
        "--raw-samples",
        default="data/diverse_reasoning_results.raw_samples.jsonl",
        help="Path to raw_samples JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/failed_samples.jsonl",
        help="Output path for failed samples (default: data/failed_samples.jsonl)",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=85.0,
        help="F1 threshold. Records with max_f1 < this are 'failed' (default: 85.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel smatchpp workers (default: auto)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(
        raw_path=Path(args.raw_samples),
        output_path=Path(args.output),
        f1_threshold=args.f1_threshold,
        n_workers=args.workers,
    )
