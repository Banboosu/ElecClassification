"""Read-only static audit of user-rejected legacy files; NEVER import/run them.

Checks named imports/path literals, distinctive APIs/output names, exact nontrivial
function AST copies and contiguous top-level plot blocks. Similarity is a review
lead, not proof of historical authorship or the direction of copying.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path("configs/legacy_source_policy.json")
SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".toml"}
MARKERS = (
    "preprocess_data_patent", "build_trend_encoder", "NormalFeatureUpdateCallback",
    "AdaptiveThreshold", "calculate_if_score", "patent_trend_encoder.h5",
    "patent_isolation_forest.pkl", "patent_model_metrics.xlsx", "patent_fault_diagnosis_results.csv",
    "tcn_training_raw_curves.png",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(root: Path, suffixes: set[str]):
    if not root.exists():
        return
    for directory, names, files in os.walk(root):
        names[:] = sorted(n for n in names if n not in {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"})
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix in suffixes:
                yield path


def is_rejected_reference(value: str, rejected: list[str]) -> bool:
    normalized = value.replace("\\", "/")
    return any(Path(path).name in normalized or normalized in {
        Path(path).stem, "legacy." + Path(path).stem,
    } for path in rejected)


def python_references(text: str, rejected: list[str]) -> list[dict]:
    tree = ast.parse(text)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values = [(alias.name, "import") for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            values = [(node.module or "", "import")]
            values += [(f"{node.module}.{alias.name}" if node.module else alias.name, "import") for alias in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            values = [(node.value, "string/path")]
        else:
            continue
        for value, kind in values:
            if is_rejected_reference(value, rejected):
                hits.append({"line": node.lineno, "kind": kind, "value": value[:250]})
    return hits


def function_fingerprints(text: str) -> list[tuple[str, int, int, str]]:
    result = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        normalized = copy.deepcopy(node)
        normalized.name = "_function_"
        if normalized.body and isinstance(normalized.body[0], ast.Expr) and isinstance(normalized.body[0].value, ast.Constant) and isinstance(normalized.body[0].value.value, str):
            normalized.body = normalized.body[1:]
        count = sum(1 for _ in ast.walk(normalized))
        if count >= 60:
            result.append((node.name, node.lineno, count, ast.dump(normalized, include_attributes=False)))
    return result


def plot_blocks(text: str) -> list[tuple[int, str]]:
    body = [node for node in ast.parse(text).body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    result = []
    for index in range(len(body) - 2):
        block = ast.Module(body=body[index:index + 3], type_ignores=[])
        if sum(1 for _ in ast.walk(block)) >= 40:
            result.append((body[index].lineno, ast.dump(block, include_attributes=False)))
    return result


def audit(root: Path, *, include_artifacts: bool) -> dict:
    policy = json.loads((root / POLICY).read_text(encoding="utf-8"))
    rejected = policy["rejected_sources"]
    inspection = set(policy["read_only_inspection_tools"])
    sources = {}
    source_functions = {}
    source_blocks = {}
    for relative in rejected:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Rejected source must be present for text-only audit: {path}")
        text = path.read_text(encoding="utf-8")
        sources[relative] = {"sha256": sha(path), "lines": len(text.splitlines())}
        for name, line, count, fingerprint in function_fingerprints(text):
            source_functions.setdefault(fingerprint, []).append({"source": relative, "name": name, "line": line, "nodes": count})
        for line, fingerprint in plot_blocks(text):
            source_blocks.setdefault(fingerprint, []).append({"source": relative, "line": line})
    files = [p for base in ("src", "scripts", "tests", "configs") for p in walk(root / base, SUFFIXES)]
    if (root / "pyproject.toml").exists():
        files.append(root / "pyproject.toml")
    code_hits, markers, copies, blocks = [], [], [], []
    scanned = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        if relative in inspection or relative == POLICY.as_posix():
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8")
        references = python_references(text, rejected) if path.suffix == ".py" else [
            {"line": i, "kind": "text/path", "value": line[:250]}
            for i, line in enumerate(text.splitlines(), 1) if is_rejected_reference(line, rejected)
        ]
        code_hits.extend({"path": relative, **hit} for hit in references)
        for index, line in enumerate(text.splitlines(), 1):
            for marker in MARKERS:
                if marker in line:
                    markers.append({"path": relative, "line": index, "marker": marker})
        if path.suffix == ".py":
            for name, line, _, fingerprint in function_fingerprints(text):
                if fingerprint in source_functions:
                    copies.append({"path": relative, "function": name, "line": line, "matches": source_functions[fingerprint]})
            for line, fingerprint in plot_blocks(text):
                if fingerprint in source_blocks:
                    blocks.append({"path": relative, "line": line, "matches": source_blocks[fingerprint]})
    related_legacy = []
    for path in walk(root / "legacy", {".py"}):
        relative = path.relative_to(root).as_posix()
        if relative in rejected:
            continue
        text = path.read_text(encoding="utf-8")
        matching = [{"line": line, "matches": source_blocks[fingerprint]} for line, fingerprint in plot_blocks(text) if fingerprint in source_blocks]
        if matching:
            related_legacy.append({"path": relative, "matching_three_statement_blocks": len(matching), "examples": matching[:3]})
    docs = []
    for path in [root / "README.md", *walk(root / "docs", {".md"})]:
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(Path(p).name in line for p in rejected):
                docs.append({"path": path.relative_to(root).as_posix(), "line": index, "text": line[:500]})
    archived_refs = []
    metadata = []
    frozen_count = 0
    artifact_paths = []
    if include_artifacts:
        for name in ("imports", "improvements", "deployments"):
            artifact_paths.extend(walk(root / "artifacts" / name, {".py", ".md", ".json", ".sh"}))
        provenance = root / "artifacts/deployment_provenance.json"
        if provenance.exists():
            artifact_paths.append(provenance)
        for path in artifact_paths:
            # Audit historical material, not this tool's own analysis outputs.
            text = path.read_text(encoding="utf-8")
            hits = [{"line": i, "text": line[:350]} for i, line in enumerate(text.splitlines(), 1)
                    if any(Path(p).name in line for p in rejected)]
            if hits:
                archived_refs.append({"path": path.relative_to(root).as_posix(), "hits": hits[:8]})
            if path.name == "frozen_protocol.json":
                frozen_count += 1
                def visit(value, field="$"):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if is_rejected_reference(str(key), rejected):
                                metadata.append({"path": path.relative_to(root).as_posix(), "field": field, "value": str(key)})
                            visit(child, field + "." + str(key))
                    elif isinstance(value, list):
                        for index, child in enumerate(value):
                            visit(child, field + f"[{index}]")
                    elif isinstance(value, str) and is_rejected_reference(value, rejected):
                        metadata.append({"path": path.relative_to(root).as_posix(), "field": field, "value": value})
                visit(json.loads(text))
    return {"read_only": True, "legacy_executed": False, "models_run": False, "test_scored": False,
            "rejected_sources": sources, "active_files_scanned": scanned,
            "historical_text_files_scanned": len(artifact_paths), "frozen_protocols_scanned": frozen_count,
            "active_named_references": code_hits, "distinctive_api_or_output_hits": markers,
            "exact_nontrivial_function_copies": copies, "exact_top_level_plot_blocks": blocks,
            "related_other_legacy": related_legacy, "documentation_mentions": docs,
            "archived_mentions": archived_refs, "frozen_input_references": metadata,
            "limitations": ["Static names and exact AST matching cannot exclude manual paraphrasing or establish historical authorship.",
                            "A shared generic plotting block does not prove the direction of copying.",
                            "min_length=18 is also present in the rejected script; its independent business provenance remains unconfirmed, so do not claim 18 points=7.5 minutes or a patent rule."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--include-artifacts", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/rejected_legacy_sources_20261029.json"))
    args = parser.parse_args()
    result = audit(args.root, include_artifacts=args.include_artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in result.items() if k != "limitations"}, ensure_ascii=False, indent=2))
