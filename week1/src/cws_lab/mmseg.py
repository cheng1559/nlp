from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import argparse

from .data import ICWB2Corpus, read_segmented_file


def build_dictionary(training_files: Sequence[Path], min_freq: int = 1) -> tuple[set[str], int]:
    counter: Counter[str] = Counter()
    for path in training_files:
        for words in read_segmented_file(path):
            for word in words:
                counter[word] += 1
    vocab = {w for w, c in counter.items() if c >= min_freq}
    max_len = max((len(w) for w in vocab), default=1)
    return vocab, max_len


@dataclass
class MaximumMatcher:
    vocab: set[str]
    max_len: int

    def forward(self, text: str) -> List[str]:
        tokens: List[str] = []
        idx = 0
        while idx < len(text):
            end = min(len(text), idx + self.max_len)
            candidate = None
            for j in range(end, idx, -1):
                piece = text[idx:j]
                if piece in self.vocab:
                    candidate = piece
                    break
            if candidate is None:
                candidate = text[idx:end]
            tokens.append(candidate)
            idx += len(candidate)
        return tokens

    def backward(self, text: str) -> List[str]:
        tokens: List[str] = []
        idx = len(text)
        while idx > 0:
            start = max(0, idx - self.max_len)
            candidate = None
            for j in range(start, idx):
                piece = text[j:idx]
                if piece in self.vocab:
                    candidate = piece
                    break
            if candidate is None:
                candidate = text[start:idx]
            tokens.append(candidate)
            idx -= len(candidate)
        tokens.reverse()
        return tokens

    def bidirectional(self, text: str) -> List[str]:
        forward = self.forward(text)
        backward = self.backward(text)
        if len(forward) != len(backward):
            return forward if len(forward) < len(backward) else backward
        forward_oov = sum(1 for token in forward if token not in self.vocab)
        backward_oov = sum(1 for token in backward if token not in self.vocab)
        if forward_oov != backward_oov:
            return forward if forward_oov < backward_oov else backward
        return forward

    def segment(self, text: str) -> List[str]:
        return self.bidirectional(text)


def segment_with_matcher(
    root: Path,
    corpus: str = "pku",
    output_dir: Path | None = None,
    min_freq: int = 1,
) -> Path:
    corpus_paths = ICWB2Corpus(root, corpus)
    vocab, max_len = build_dictionary(
        [corpus_paths.training_path()], min_freq=min_freq)
    matcher = MaximumMatcher(vocab=vocab, max_len=max_len)
    raw_test = corpus_paths.raw_test_path()
    lines = raw_test.read_text(encoding="utf-8").splitlines()
    segmented: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            segmented.append("")
            continue
        tokens = matcher.segment(line)
        segmented.append(" ".join(tokens))
    output_dir = output_dir or (root / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{corpus}_mmseg.txt"
    out_path.write_text("\n".join(segmented), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run maximum matching segmentation")
    parser.add_argument("--data-root", type=str, default=str(default_root),
                        help="Project root containing data directory")
    parser.add_argument("--corpus", type=str, default="pku",
                        choices=["pku", "msr", "as", "cityu"], help="Corpus name")
    parser.add_argument("--min-freq", type=int, default=1,
                        help="Minimum dictionary frequency")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out = segment_with_matcher(
        root=Path(args.data_root),
        corpus=args.corpus,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        min_freq=args.min_freq,
    )
    print(out)


if __name__ == "__main__":
    main()
