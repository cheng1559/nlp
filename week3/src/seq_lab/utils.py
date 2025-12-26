from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import regex as re

TOKEN_RE = re.compile(r"\p{Han}+|[a-zA-Z]+|\d+|[\p{P}\p{S}]")


def simple_tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text) if tok.strip()]


def read_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line
