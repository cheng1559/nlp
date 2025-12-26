from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .data import CWSDataModule, ICWB2Corpus
from .model import BiLSTMTagger, train_model


def run_training(args):
    data_root = Path(args.data_root)
    checkpoint = train_model(
        data_root=data_root,
        corpus=args.corpus,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        lr=args.lr,
        limit=args.limit,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    if args.no_segment:
        return checkpoint
    data = CWSDataModule(data_root=data_root,
                         corpus=args.corpus, batch_size=1, val_ratio=0.01)
    data.prepare_data()
    data.setup()
    model = BiLSTMTagger.load_from_checkpoint(checkpoint, id_to_char=data.itos)
    model.eval()
    corpus_paths = ICWB2Corpus(data_root, args.corpus)
    raw_test = corpus_paths.raw_test_path()
    lines = raw_test.read_text(encoding="utf-8").splitlines()
    outputs = []
    for line in tqdm(lines, desc="segment"):
        stripped = line.strip()
        if not stripped:
            outputs.append("")
            continue
        tokens = model.segment(stripped, data.stoi)
        outputs.append(" ".join(tokens))
    out_dir = Path(args.output_dir) if args.output_dir else (
        data_root / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.corpus}_bilstm.txt"
    out_path.write_text("\n".join(outputs), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train BiLSTM CWS model")
    parser.add_argument("--data-root", type=str, default=str(default_root),
                        help="Project root containing data directory")
    parser.add_argument("--corpus", type=str, default="pku",
                        choices=["pku", "msr", "as", "cityu"], help="Corpus name")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional sentence cap for debugging")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-segment", action="store_true",
                        help="Skip test segmentation after training")
    return parser.parse_args()


def main():
    args = parse_args()
    path = run_training(args)
    print(path)


if __name__ == "__main__":
    main()
