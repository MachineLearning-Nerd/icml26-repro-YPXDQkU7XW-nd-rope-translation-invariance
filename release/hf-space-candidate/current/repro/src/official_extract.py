"""Load selected top-level functions from a pinned source file via its AST.

The image implementation imports a full legacy DeiT stack.  Extracting only the
four mathematical functions lets us execute the authors' code verbatim without
silently patching the model or its dependencies.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def load_functions(path: Path, names: Iterable[str]) -> dict[str, object]:
    requested = set(names)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in requested
    ]
    found = {node.name for node in selected}
    if found != requested:
        raise KeyError(f"missing functions: {sorted(requested - found)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"torch": torch, "np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in requested}
