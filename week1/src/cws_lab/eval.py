from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .model import SequenceMetrics, spans_from_tokens


def evaluate_lines(pred_lines: Sequence[str], gold_lines: Sequence[str]) -> SequenceMetrics:
    # Trim trailing blank lines to reduce benign length mismatches
    while pred_lines and not pred_lines[-1].strip():
        pred_lines = pred_lines[:-1]
    while gold_lines and not gold_lines[-1].strip():
        gold_lines = gold_lines[:-1]
    max_len = min(len(pred_lines), len(gold_lines))
    pred_lines = pred_lines[:max_len]
    gold_lines = gold_lines[:max_len]
    total_tp = 0
    total_pred = 0
    total_gold = 0
    for pred, gold in zip(pred_lines, gold_lines):
        pred_tokens = pred.strip().split() if pred.strip() else []
        gold_tokens = gold.strip().split() if gold.strip() else []
        pred_spans = spans_from_tokens(pred_tokens)
        gold_spans = spans_from_tokens(gold_tokens)
        total_tp += len(pred_spans & gold_spans)
        total_pred += len(pred_spans)
        total_gold += len(gold_spans)
    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall > 0 else 0.0
    return SequenceMetrics(precision, recall, f1)


def evaluate_files(pred_path: Path, gold_path: Path) -> SequenceMetrics:
    pred_lines = pred_path.read_text(encoding="utf-8").splitlines()
    gold_lines = gold_path.read_text(encoding="utf-8").splitlines()
    return evaluate_lines(pred_lines, gold_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmented output against gold file")
    parser.add_argument("pred", type=str, help="Predicted segmented file")
    parser.add_argument("gold", type=str, help="Gold segmented file")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = evaluate_files(Path(args.pred), Path(args.gold))
    print(f"P={metrics.precision:.4f} R={metrics.recall:.4f} F1={metrics.f1:.4f}")


if __name__ == "__main__":
    main()
