"""
Utility functions for AMR benchmark evaluation.
"""

import json
import penman
from typing import Optional, Tuple, Dict, Any

from smatchpp import Smatchpp, solvers
from smatchpp.formalism.amr import tools as amrtools


def remove_wiki_from_amr(amr_str: str) -> str:
    """
    Remove triples with :wiki role from AMR string.
    
    Args:
        amr_str: The AMR graph string.
        
    Returns:
        The AMR string with all :wiki relations removed.
    """
    try:
        g = penman.decode(amr_str)
        # Filter out triples with :wiki role
        new_triples = [t for t in g.triples if t[1] != ':wiki']
        # Create a new graph with the filtered triples
        new_g = penman.Graph(new_triples)
        # Encode back to string
        return penman.encode(new_g)
    except Exception as e:
        print(f"Error removing wiki: {e}")
        return amr_str


def fix_amr_parentheses(amr_str: str) -> str:
    """
    Fix unbalanced parentheses in AMR string by adding missing closing parentheses.
    
    Args:
        amr_str: The AMR graph string that may have unbalanced parentheses.
        
    Returns:
        The AMR string with balanced parentheses.
    """
    if not amr_str:
        return amr_str
    
    open_count = amr_str.count('(')
    close_count = amr_str.count(')')
    
    if open_count > close_count:
        # Add missing closing parentheses
        amr_str = amr_str + ')' * (open_count - close_count)
    
    return amr_str


def extract_amr_from_output(output: str) -> Optional[str]:
    """
    Extract AMR graph from model output by finding the first '(' character.
    
    Args:
        output: The raw model output string.
        
    Returns:
        The extracted AMR string or None if no AMR found.
    """
    amr_start = output.find('(')
    if amr_start != -1:
        return output[amr_start:].strip()
    return None


def benchmark_smatch(
    jsonl_path: str,
    ground_truth_field: str,
    predict_field: str,
    remove_wiki: bool = True,
    extract_amr_from_predict: bool = True
) -> Tuple[Dict[str, Any], Any]:
    """
    Run SMATCH benchmark on a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file containing both predictions and ground truth.
        ground_truth_field: Field name for ground truth AMR in each JSON line.
        predict_field: Field name for predicted AMR in each JSON line.
        remove_wiki: Whether to remove :wiki annotations before scoring (default: True).
        extract_amr_from_predict: Whether to extract AMR from predict field 
                                   (find first '(' character). Default: True.
    
    Returns:
        A tuple of (score_dict, optimization_status) from smatchpp.
    
    Example:
        >>> score, status = benchmark_smatch(
        ...     'inference_output.jsonl',
        ...     ground_truth_field='amr',
        ...     predict_field='output_model'
        ... )
        >>> print(json.dumps(score, indent=2))
    """
    # Load data
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    predicted_graphs = []
    reference_graphs = []
    
    for item in data:
        # Get predicted AMR
        predict_value = item.get(predict_field, '')
        if extract_amr_from_predict:
            predicted_amr = extract_amr_from_output(predict_value)
        else:
            predicted_amr = predict_value if predict_value else None
        
        if predicted_amr is None:
            # Skip items without valid prediction
            continue
            
        # Get ground truth AMR
        ground_truth_amr = item.get(ground_truth_field, '')
        if not ground_truth_amr:
            continue
        
        # Remove wiki if requested
        if remove_wiki:
            predicted_amr = remove_wiki_from_amr(predicted_amr)
            ground_truth_amr = remove_wiki_from_amr(ground_truth_amr)
        
        predicted_graphs.append(predicted_amr)
        reference_graphs.append(ground_truth_amr)
    
    print(f"Loaded {len(predicted_graphs)} valid samples for evaluation")
    
    # Setup SMATCH scorer
    graph_standardizer = amrtools.AMRStandardizer()
    ilp = solvers.ILP()
    measure = Smatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer)
    
    # Score corpus
    score, optimization_status = measure.score_corpus(predicted_graphs, reference_graphs)
    
    return score, optimization_status


def benchmark_smatch_from_files(
    predict_jsonl_path: str,
    ground_truth_jsonl_path: str,
    predict_field: str,
    ground_truth_field: str,
    remove_wiki: bool = True,
    extract_amr_from_predict: bool = True
) -> Tuple[Dict[str, Any], Any]:
    """
    Run SMATCH benchmark using separate JSONL files for predictions and ground truth.
    
    Args:
        predict_jsonl_path: Path to the JSONL file containing predictions.
        ground_truth_jsonl_path: Path to the JSONL file containing ground truth.
        predict_field: Field name for predicted AMR in predict JSONL.
        ground_truth_field: Field name for ground truth AMR in ground truth JSONL.
        remove_wiki: Whether to remove :wiki annotations before scoring (default: True).
        extract_amr_from_predict: Whether to extract AMR from predict field.
    
    Returns:
        A tuple of (score_dict, optimization_status) from smatchpp.
    """
    # Load prediction data
    with open(predict_jsonl_path, 'r', encoding='utf-8') as f:
        predict_data = [json.loads(line) for line in f]
    
    # Load ground truth data
    with open(ground_truth_jsonl_path, 'r', encoding='utf-8') as f:
        ground_truth_data = [json.loads(line) for line in f]
    
    predicted_graphs = []
    reference_graphs = []
    
    for pred_item, gt_item in zip(predict_data, ground_truth_data):
        # Get predicted AMR
        predict_value = pred_item.get(predict_field, '')
        if extract_amr_from_predict:
            predicted_amr = extract_amr_from_output(predict_value)
        else:
            predicted_amr = predict_value if predict_value else None
        
        if predicted_amr is None:
            continue
            
        # Get ground truth AMR
        ground_truth_amr = gt_item.get(ground_truth_field, '')
        if not ground_truth_amr:
            continue
        
        # Remove wiki if requested
        if remove_wiki:
            predicted_amr = remove_wiki_from_amr(predicted_amr)
            ground_truth_amr = remove_wiki_from_amr(ground_truth_amr)
        
        predicted_graphs.append(predicted_amr)
        reference_graphs.append(ground_truth_amr)
    
    print(f"Loaded {len(predicted_graphs)} valid samples for evaluation")
    
    # Setup SMATCH scorer
    graph_standardizer = amrtools.AMRStandardizer()
    ilp = solvers.ILP()
    measure = Smatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer)
    
    # Score corpus
    score, optimization_status = measure.score_corpus(predicted_graphs, reference_graphs)
    
    return score, optimization_status


def add_smatch_scores_to_jsonl(
    jsonl_path: str,
    gold_field: str = "gold_amr",
    predict_field: str = "output_amr",
    score_field: str = "smatch_f1",
    output_path: Optional[str] = None,
    remove_wiki: bool = True,
    extract_amr_from_predict: bool = False
) -> None:
    """
    Calculate per-sample Smatch F1 scores and add them to each entry in the JSONL file.
    
    Args:
        jsonl_path: Path to the input JSONL file.
        gold_field: Field name for gold AMR (default: "gold_amr").
        predict_field: Field name for predicted AMR (default: "output_amr").
        score_field: Field name for the F1 score to add (default: "smatch_f1").
        output_path: Path to write the output. If None, overwrites the input file.
        remove_wiki: Whether to remove :wiki annotations before scoring (default: True).
        extract_amr_from_predict: Whether to extract AMR from predict field.
    """
    # Import AMRHint for remove_wiki_from_amr
    from .amr_hint import AMRHint
    
    # Load data
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} entries from {jsonl_path}")
    
    # Setup SMATCH scorer
    graph_standardizer = amrtools.AMRStandardizer()
    ilp = solvers.ILP()
    measure = Smatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer)
    
    processed = 0
    skipped = 0
    
    for i, item in enumerate(data):
        # Get predicted AMR
        predict_value = item.get(predict_field, '')
        if extract_amr_from_predict:
            predicted_amr = extract_amr_from_output(predict_value)
        else:
            predicted_amr = predict_value if predict_value else None
        
        # Get gold AMR
        gold_amr = item.get(gold_field, '')
        
        # Skip if either is missing
        if not predicted_amr or not gold_amr:
            item[score_field] = None
            skipped += 1
            continue
        
        # Skip failed entries
        if item.get('success') == False:
            item[score_field] = None
            skipped += 1
            continue
        
        try:
            # Fix unbalanced parentheses in predicted AMR
            predicted_amr = fix_amr_parentheses(predicted_amr)
            
            # Save the fixed output_amr back to the item
            item[predict_field] = predicted_amr
            
            # Remove wiki if requested (using AMRHint's method)
            predicted_amr_clean = predicted_amr
            gold_amr_clean = gold_amr
            if remove_wiki:
                predicted_amr_clean = AMRHint.remove_wiki_from_amr(predicted_amr)
                gold_amr_clean = AMRHint.remove_wiki_from_amr(gold_amr)
            
            # Calculate Smatch score for this single pair
            score, _ = measure.score_corpus([predicted_amr_clean], [gold_amr_clean])
            
            # Extract F1 score (main Smatch F1)
            # Structure: score['main']['F1']['result'] -> np.float64
            f1 = float(score.get('main', {}).get('F1', {}).get('result', 0.0))
            item[score_field] = round(f1, 4)
            processed += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(data)} entries...")
                
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            item[score_field] = None
            skipped += 1
    
    # Write output
    out_path = output_path if output_path else jsonl_path
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nDone! Processed {processed} entries, skipped {skipped}")
    print(f"Results written to {out_path}")
    
    # Calculate average F1
    valid_scores = [item[score_field] for item in data if item.get(score_field) is not None]
    if valid_scores:
        avg_f1 = sum(valid_scores) / len(valid_scores)
        print(f"Average F1 Score: {avg_f1:.4f}")


# CLI usage example
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AMR Benchmark with SMATCH')
    parser.add_argument('--input', required=True, help='Path to JSONL file')
    parser.add_argument('--ground-truth-field', required=True, help='Field name for ground truth AMR')
    parser.add_argument('--predict-field', required=True, help='Field name for predicted AMR')
    parser.add_argument('--no-remove-wiki', action='store_true', help='Do not remove wiki annotations')
    parser.add_argument('--no-extract-amr', action='store_true', 
                        help='Do not extract AMR from predict field (use as-is)')
    
    args = parser.parse_args()
    
    score, status = benchmark_smatch(
        jsonl_path=args.input,
        ground_truth_field=args.ground_truth_field,
        predict_field=args.predict_field,
        remove_wiki=not args.no_remove_wiki,
        extract_amr_from_predict=not args.no_extract_amr
    )
    
    print("\nSMATCH F1 Score:")
    print(json.dumps(score, indent=2))
