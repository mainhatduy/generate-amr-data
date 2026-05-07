import penman
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Set


class FindFrame:
    """Find all instances of a given frame in an AMR graph."""

    def __init__(self, frames_path=None):
        if frames_path is None:
            # Default to bundled PropBank frames under concepts/
            frames_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "concepts",
                "propbank_amr_frames.jsonl",
            )
        self.frames = self.load_frames(frames_path)
        
    def convert_roleset_id(self, roleset_id: str) -> str:
        """Chuyển đổi roleset_id từ định dạng có dấu chấm sang dấu gạch ngang."""
        return roleset_id.replace('.', '-').replace('_', '-')

    def load_frames(self, frames_path: str) -> Dict[str, Dict]:
        """Load frames from jsonl file.
        
        Returns:
            Dict mapping roleset_id -> frame dict
        """
        frames = {}
        if os.path.exists(frames_path):
            with open(frames_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # data structure: {"frame": "abandon-01", "meaning": "...", "arguments": {"ARG0": "..."}}
                        frames[data["frame"]] = data
                    except json.JSONDecodeError:
                        continue
        return frames
    
    def get_frame(self, roleset_id: str) -> Optional[Dict]:
        """Get frame data by roleset_id."""
        roleset_id = self.convert_roleset_id(roleset_id)
        return self.frames.get(roleset_id, None)
    
    def get_frames_by_name(self, frame_name: str) -> List[Dict]:
        """Get all frames with the same base name.
        
        For example, get_frames_by_name("want") returns all frames like want-01, want-02, etc.
        
        Args:
            frame_name: Base name of the frame (without -NN suffix)
            
        Returns:
            List of frame data dicts for all matching frames
        """
        frame_name = frame_name.replace('.', '-').replace('_', '-')
        matching_frames = []
        for frame_id, frame_data in self.frames.items():
            # Check if frame_id starts with frame_name followed by -
            if frame_id.startswith(f"{frame_name}-"):
                matching_frames.append(frame_data)
        # Sort by frame ID for consistent ordering
        matching_frames.sort(key=lambda x: x.get('frame', ''))
        return matching_frames
    
    def llm_convenient_format(self, roleset_id: str, args: Optional[List[str]] = None) -> str:
        """Return a convenient string format for LLM."""
        frame_data = self.get_frame(roleset_id)
        if frame_data is None:
            return f"Frame '{roleset_id}' not found."
        
        lines = [
            f"Frame: {frame_data['frame']}",
            f"Meaning: {frame_data['meaning']}",
            "Arguments:"
        ]
        
        arguments = frame_data.get("arguments", {})
        # Sort keys to have consistent output
        sorted_args = sorted(arguments.keys())
        
        for arg_key in sorted_args:
            lines.append(f"  - {arg_key}: {arguments[arg_key]}")
        
        return "\n".join(lines)

    def llm_convenient_format_batch(self, roleset_ids: List[str]) -> str:
        results = []
        for rid in roleset_ids:
            results.append(self.llm_convenient_format(rid))
        return "\n\n".join(results)


class FindConcept:
    """Find information about special concepts."""
    
    def __init__(self, concepts_path=None):
        if concepts_path is None:
            concepts_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "concepts",
                "concept_groups.json",
            )
        self.concepts_data, self.concept_mapping = self.load_concepts(concepts_path)
        
    def load_concepts(self, concepts_path: str) -> Tuple[Dict, Dict]:
        """Load concepts from json file."""
        data = {}
        mapping = {} # maps concept/alias -> group_name
        
        if os.path.exists(concepts_path):
            with open(concepts_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for group, info in data.items():
                        mapping[group] = group
                        aliases = info.get("aliases", [])
                        for alias in aliases:
                            mapping[alias] = group
                except json.JSONDecodeError:
                    print(f"Error decoding {concepts_path}")
        return data, mapping

    def llm_convenient_format(self, concept_name: str) -> str:
        """Return information about a concept if it exists in our knowledge base."""
        group = self.concept_mapping.get(concept_name)
        if not group:
            return ""
        
        info = self.concepts_data.get(group)
        if not info:
            return ""
            
        lines = []
        lines.append(f"Concept: {concept_name}")
        if group != concept_name:
            lines.append(f"Type: {group}")
            
        if info.get("relation_pointer"):
            lines.append(f"Relation Pointer: {info['relation_pointer']}")
            
        return "\n".join(lines)


class AMRHint:
    """Class to add hints from PropBank frames and Concept definitions into AMR graph."""
    
    def __init__(self, frames_path=None, concepts_path=None):
        self.frame_finder = FindFrame(frames_path)
        self.concept_finder = FindConcept(concepts_path)
    
    @staticmethod
    def remove_wiki_from_amr(amr_str: str) -> str:
        """Remove triples with :wiki role from AMR string."""
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
    
    def extract_frames_and_concepts(self, amr_input) -> Tuple[Dict[str, Dict], List[str]]:
        """Extract frames (with used args) and concepts from AMR string or Graph.
        
        Only returns concepts that exist in concept_groups.json.
        """
        if isinstance(amr_input, penman.Graph):
            g = amr_input
        else:
            try:
                g = penman.decode(amr_input)
            except Exception:
                return {}, []
                
        triples = g.triples
        
        frames_info = {} # mapping variable -> {"frame": name, "args": [list of arg numbers]}
        concepts_found = set()
        
        # 1. First pass: Identify frames and concepts from instances
        # Instance triple: (variable, :instance, concept/frame)
        var_to_type = {} 
        
        for source, role, target in triples:
            if role == ':instance':
                var_to_type[source] = target
                # Check if it looks like a frame (has -number suffix)
                # Regex for frame: ends with dash followed by digits
                if re.match(r'.+-\d+$', target):
                    frames_info[source] = {"frame": target, "args": []}
                else:
                    # Only add concept if it exists in concept_groups.json
                    if target in self.concept_finder.concept_mapping:
                        concepts_found.add(target)
        
        # 2. Second pass: Identify arguments used for the frames
        for source, role, target in triples:
            if source in frames_info:
                # Check if role matches :ARGn
                match = re.match(r':ARG(\d+)', role)
                if match:
                    arg_num = match.group(1)
                    if arg_num not in frames_info[source]["args"]:
                        frames_info[source]["args"].append(arg_num)
        
        return frames_info, list(concepts_found)
    
    def get_hints(self, amr_input) -> str:
        """Generate hints for LLM from AMR string or Graph."""
        frames_info, concepts = self.extract_frames_and_concepts(amr_input)
        
        hints = []
        
        # Add frame hints
        for var, info in frames_info.items():
            frame_name = info["frame"]
            args = info["args"]
            hint = self.frame_finder.llm_convenient_format(frame_name, args=args)
            if hint and not hint.endswith("not found."):
                hints.append(hint)
        
        # Add concept hints
        for concept in concepts:
            hint = self.concept_finder.llm_convenient_format(concept)
            if hint:
                hints.append(hint)
        
        # Deduplicate hints (in case multiple variables use same frame/concept)
        unique_hints = []
        seen_hints = set()
        for h in hints:
            if h not in seen_hints:
                unique_hints.append(h)
                seen_hints.add(h)
                
        return "\n\n".join(unique_hints)
    
    def get_hints_partial(self, amr_input, percentage: float = 1.0) -> str:
        """
        Generate partial hints for LLM based on percentage.
        
        Args:
            amr_input: AMR string or graph
            percentage: Fraction of hints to include (0.0-1.0). Root frame is always included.
            
        Returns:
            Formatted hints string
        """
        import random
        
        if percentage >= 1.0:
            return self.get_hints(amr_input)
        
        frames_info, concepts = self.extract_frames_and_concepts(amr_input)
        
        # Get root variable (first variable in the AMR)
        if isinstance(amr_input, penman.Graph):
            g = amr_input
        else:
            try:
                g = penman.decode(amr_input)
            except Exception:
                return ""
        
        root_var = g.top
        
        all_hints = []
        root_hint = None
        
        # Process frame hints, keeping root separate
        for var, info in frames_info.items():
            frame_name = info["frame"]
            args = info["args"]
            hint = self.frame_finder.llm_convenient_format(frame_name, args=args)
            if hint and not hint.endswith("not found."):
                if var == root_var:
                    root_hint = hint
                else:
                    all_hints.append(hint)
        
        # Process concept hints
        for concept in concepts:
            hint = self.concept_finder.llm_convenient_format(concept)
            if hint:
                all_hints.append(hint)
        
        # Calculate how many additional hints to include
        total_available = len(all_hints)
        num_to_include = int(total_available * percentage)
        
        # Randomly select hints (excluding root which is always included)
        if num_to_include > 0 and total_available > 0:
            selected_hints = random.sample(all_hints, min(num_to_include, total_available))
        else:
            selected_hints = []
        
        # Build final list: root first, then selected
        final_hints = []
        if root_hint:
            final_hints.append(root_hint)
        final_hints.extend(selected_hints)
        
        return "\n\n".join(final_hints)

    def get_hints_structured(
        self,
        amr_input,
        *,
        include_all_arguments: bool = True,
    ) -> Dict[str, Any]:
        """Generate structured hints (Python dict) from an AMR string or Graph.

        Output shape:
            {
              "root": "want-01" | None,
              "frames": [
                {
                  "frame": "want-01",
                  "meaning": "want, desire",
                  "used_args": ["ARG0", "ARG1"],
                  "used_arguments": {"ARG0": "...", "ARG1": "..."},
                  "arguments": {"ARG0": "...", ...}  # optional
                },
                ...
              ],
              "concepts": [
                {"concept": "...", "type": "..." | None, "relation_pointer": "..." | None},
                ...
              ]
            }

        Notes:
        - Requires AMR input to already be in Penman format; this module does not infer frames from raw text.
        - Frames/concepts are de-duplicated.
        """
        frames_info, concepts = self.extract_frames_and_concepts(amr_input)

        root: Optional[str] = None
        try:
            g = amr_input if isinstance(amr_input, penman.Graph) else penman.decode(amr_input)
            root_var = g.top
            if root_var in frames_info:
                root = frames_info[root_var].get("frame")
        except Exception:
            root = None

        frames: List[Dict[str, Any]] = []
        seen_frames: Set[str] = set()

        for _var, info in frames_info.items():
            frame_name = info.get("frame")
            if not frame_name:
                continue

            frame_data = self.frame_finder.get_frame(frame_name)
            if not frame_data:
                continue

            frame_id = frame_data.get("frame", frame_name)
            if frame_id in seen_frames:
                continue

            arguments: Dict[str, str] = dict(frame_data.get("arguments", {}) or {})
            used_nums = sorted(set(info.get("args", []) or []), key=lambda x: int(x) if str(x).isdigit() else 10**9)
            used_args = [f"ARG{n}" for n in used_nums]
            used_arguments = {k: arguments[k] for k in used_args if k in arguments}

            frame_obj: Dict[str, Any] = {
                "frame": frame_id,
                "meaning": frame_data.get("meaning", ""),
                "used_args": used_args,
                "used_arguments": used_arguments,
            }
            if include_all_arguments:
                frame_obj["arguments"] = arguments

            frames.append(frame_obj)
            seen_frames.add(frame_id)

        concept_out: List[Dict[str, Any]] = []
        seen_concepts: Set[str] = set()
        for concept in concepts:
            if concept in seen_concepts:
                continue
            seen_concepts.add(concept)

            group = self.concept_finder.concept_mapping.get(concept)
            info = self.concept_finder.concepts_data.get(group) if group else None
            if not info:
                continue

            concept_obj: Dict[str, Any] = {"concept": concept}
            concept_obj["type"] = group if group and group != concept else None
            concept_obj["relation_pointer"] = info.get("relation_pointer")
            concept_out.append(concept_obj)

        return {
            "root": root,
            "frames": frames,
            "concepts": concept_out,
        }

    def get_hints_json(
        self,
        amr_input,
        *,
        include_all_arguments: bool = True,
        indent: Optional[int] = 4,
    ) -> str:
        """Generate structured hints and serialize as JSON string."""
        obj = self.get_hints_structured(amr_input, include_all_arguments=include_all_arguments)
        return json.dumps(obj, ensure_ascii=False, indent=indent)