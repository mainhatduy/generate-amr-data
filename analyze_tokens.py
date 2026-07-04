#!/usr/bin/env python
"""
analyze_tokens.py — Thống kê số lượng input token và output token để tối ưu hóa tham số cho diverse_sampling_config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Default paths
DEFAULT_CONFIG_PATH = Path("configs/diverse_sampling_config.json")
DEFAULT_OUTPUT_PATH = Path("data/diverse_reasoning_results.jsonl")

# User prompt format
USER_PROMPT = (
    "Convert the following English sentence into its Abstract Meaning Representation (AMR):\n\n"
    "<sentence>{sentence}</sentence>"
)


def extract_thinking(response: str) -> str:
    """Extract thinking process before </think> tag."""
    if "</think>" in response:
        thinking = response.split("</think>", 1)[0]
        thinking = thinking.replace("<think>", "")
        return thinking.strip()
    
    if "<think>" in response:
        thinking = response.split("<think>", 1)[1]
        return thinking.strip()
        
    thinking = re.sub(r"<amr>.*?</amr>", "", response, flags=re.DOTALL)
    return thinking.strip()


def extract_amr(response: str) -> str | None:
    """Extract content between <amr>...</amr> tags."""
    match = re.search(r"<amr>(.*?)</amr>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "<amr>" in response:
        return response.split("<amr>", 1)[1].strip()
    return None


def calculate_stats(data: List[int] | np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers."""
    if not data:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "p99": 0}
    
    arr = np.array(data)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def print_stats_table(title: str, stats: Dict[str, float]):
    print(f"\n=== {title} ===")
    print(f"  Min:    {stats['min']:.1f}")
    print(f"  Max:    {stats['max']:.1f}")
    print(f"  Mean:   {stats['mean']:.1f}")
    print(f"  Median: {stats['median']:.1f}")
    print(f"  p90:    {stats['p90']:.1f}")
    print(f"  p95:    {stats['p95']:.1f}")
    print(f"  p99:    {stats['p99']:.1f}")


def _get_token_count(ids_or_dict) -> int:
    if isinstance(ids_or_dict, dict) and "input_ids" in ids_or_dict:
        return len(ids_or_dict["input_ids"])
    elif hasattr(ids_or_dict, "input_ids"):
        return len(ids_or_dict.input_ids)
    return len(ids_or_dict)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_tokens(
    config_path: Path,
    raw_samples_path: Path,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    load_dotenv()
    
    # 1. Load config and tokenizer
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)
    
    model_name = config.get("model_config", {}).get("model")
    if not model_name:
        raise ValueError("Missing model_config.model in config file.")
    
    print(f"Loading tokenizer for model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
    
    max_tokens_limit = int(config.get("sampling_config", {}).get("max_tokens", 2048))
    
    # 2. Check raw samples file
    if not raw_samples_path.exists():
        raise FileNotFoundError(f"Raw samples file not found at: {raw_samples_path}")
    
    print(f"Reading raw samples from {raw_samples_path}...")
    
    # Token count lists
    input_token_counts = []
    output_token_counts = []
    thinking_token_counts = []
    amr_token_counts = []
    total_seq_lengths = []
    
    num_truncated_responses = 0
    total_responses_count = 0
    total_sentences_count = 0
    
    with raw_samples_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            sentence = data.get("sentence")
            responses = data.get("responses", [])
            
            if not sentence or not responses:
                continue
                
            total_sentences_count += 1
            
            # Count input tokens
            user_prompt = USER_PROMPT.format(sentence=sentence)
            messages = [{"role": "user", "content": user_prompt}]
            
            # Format using chat template to match engine behavior
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            input_len = _get_token_count(input_ids)
            input_token_counts.append(input_len)
            
            # Analyze responses
            for resp in responses:
                total_responses_count += 1
                
                resp_ids = tokenizer.encode(resp)
                output_len = _get_token_count(resp_ids)
                output_token_counts.append(output_len)
                
                total_seq_lengths.append(input_len + output_len)
                
                # Check truncation conditions:
                # - Length is at the max_tokens limit
                # - Or it lacks </amr> tag
                is_missing_amr_tag = "</amr>" not in resp
                is_at_limit = output_len >= max_tokens_limit
                
                if is_missing_amr_tag or is_at_limit:
                    num_truncated_responses += 1
                
                # Extra: count thinking and amr sub-tokens
                thinking_text = extract_thinking(resp)
                amr_text = extract_amr(resp)
                
                thinking_len = _get_token_count(tokenizer.encode(thinking_text))
                thinking_token_counts.append(thinking_len)
                
                if amr_text:
                    amr_len = _get_token_count(tokenizer.encode(amr_text))
                    amr_token_counts.append(amr_len)
                else:
                    amr_token_counts.append(0)
            
            if total_sentences_count % 100 == 0:
                print(f"  Processed {total_sentences_count} sentences...")
                
    # 3. Calculate statistics
    print("\n--- Token Statistics Results ---")
    print(f"Total Sentences Analyzed: {total_sentences_count}")
    print(f"Total Generated Responses Analyzed: {total_responses_count}")
    
    input_stats = calculate_stats(input_token_counts)
    output_stats = calculate_stats(output_token_counts)
    thinking_stats = calculate_stats(thinking_token_counts)
    amr_stats = calculate_stats(amr_token_counts)
    total_seq_stats = calculate_stats(total_seq_lengths)
    
    print_stats_table("Input Prompt Tokens", input_stats)
    print_stats_table("Output Response Tokens (Total)", output_stats)
    print_stats_table("Thinking Part Tokens", thinking_stats)
    print_stats_table("AMR Part Tokens", amr_stats)
    print_stats_table("Total Sequence Tokens (Input + Output)", total_seq_stats)
    
    # 4. Truncation rate
    trunc_rate = (num_truncated_responses / total_responses_count) * 100 if total_responses_count > 0 else 0
    print(f"\n=== Truncation Analysis ===")
    print(f"  Truncated Responses: {num_truncated_responses} / {total_responses_count} ({trunc_rate:.2f}%)")
    
    # 5. Parameter Optimization Recommendations
    # We want max_model_len to cover at least 99% of total sequences (plus a small safety buffer)
    # Standard values are powers of 2 or multiples of 512
    p99_total = total_seq_stats["p99"]
    suggested_max_model_len = 512
    while suggested_max_model_len < (p99_total + 64):
        suggested_max_model_len *= 2
        
    p99_output = output_stats["p99"]
    suggested_max_tokens = 256
    while suggested_max_tokens < (p99_output + 32):
        suggested_max_tokens *= 2
        # Avoid going beyond suggested_max_model_len
        if suggested_max_tokens >= suggested_max_model_len:
            break
            
    print(f"\n=== Configuration Recommendations ===")
    print(f"  Current max_model_len: {config.get('model_config', {}).get('max_model_len', 4096)}")
    print(f"  Suggested max_model_len (99th percentile + safety buffer): {suggested_max_model_len}")
    print(f"  Current max_tokens: {max_tokens_limit}")
    print(f"  Suggested max_tokens (99th percentile + safety buffer): {suggested_max_tokens}")
    
    results = {
        "sentences": total_sentences_count,
        "responses": total_responses_count,
        "truncations": num_truncated_responses,
        "truncation_rate": trunc_rate,
        "input_stats": input_stats,
        "output_stats": output_stats,
        "thinking_stats": thinking_stats,
        "amr_stats": amr_stats,
        "total_seq_stats": total_seq_stats,
        "suggested_max_model_len": suggested_max_model_len,
        "suggested_max_tokens": suggested_max_tokens,
    }
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thống kê số lượng input token và output token để tối ưu hóa tham số"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to diverse sampling config JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--raw-samples",
        help="Path to raw samples JSONL file (default: derived from config output path)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of sentences to process",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update the config file with the recommended parameters",
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    if args.raw_samples:
        raw_samples_path = Path(args.raw_samples)
    else:
        config = load_config(config_path)
        output_path = Path(config.get("pipeline_config", {}).get("output", DEFAULT_OUTPUT_PATH))
        # Deriving the raw samples path as done in diverse_sampling_pipeline.py
        if output_path.suffix == ".jsonl":
            raw_samples_path = output_path.with_name(output_path.name.replace(".jsonl", ".raw_samples.jsonl"))
        else:
            raw_samples_path = output_path.with_name(output_path.name + ".raw_samples.jsonl")
            
    try:
        results = analyze_tokens(
            config_path=config_path,
            raw_samples_path=raw_samples_path,
            max_samples=args.max_samples,
        )
        
        if args.update_config:
            config = load_config(config_path)
            
            # Update model_config.max_model_len
            if "model_config" not in config:
                config["model_config"] = {}
            config["model_config"]["max_model_len"] = results["suggested_max_model_len"]
            
            # Update sampling_config.max_tokens
            if "sampling_config" not in config:
                config["sampling_config"] = {}
            config["sampling_config"]["max_tokens"] = results["suggested_max_tokens"]
            
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print(f"\nUpdated config file {config_path} successfully!")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
