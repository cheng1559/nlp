from __future__ import annotations

import argparse
from pathlib import Path

from .sgns import train_sgns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train skip-gram with negative sampling on zhwiki")
    parser.add_argument("--data-root", type=str, default=str(Path(__file__).resolve(
    ).parents[2]), help="Project root containing data/zhwiki-latest-pages-articles.xml.bz2")
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit-articles", type=int,
                        default=None, help="Optional article cap for debugging")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    dump_path = root / "data" / "zhwiki-latest-pages-articles.xml.bz2"
    out_dir = Path(args.output_dir) if args.output_dir else (root / "output")
    ckpt = train_sgns(
        dump_path=dump_path,
        output_dir=out_dir,
        min_freq=args.min_freq,
        max_size=args.max_size,
        window_size=args.window_size,
        negative=args.negative,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        max_epochs=args.epochs,
        limit_articles=args.limit_articles,
    )
    print(ckpt)


if __name__ == "__main__":
    main()
