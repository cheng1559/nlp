from __future__ import annotations

import argparse
from pathlib import Path

from .glove import train_glove


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GloVe on zhwiki")
    parser.add_argument("--data-root", type=str, default=str(Path(__file__).resolve(
    ).parents[2]), help="Project root containing data/zhwiki-latest-pages-articles.xml.bz2")
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--x-max", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--limit-articles", type=int,
                        default=None, help="Optional article cap for debugging")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    dump_path = root / "data" / "zhwiki-latest-pages-articles.xml.bz2"
    out_dir = Path(args.output_dir) if args.output_dir else (root / "output")
    ckpt = train_glove(
        dump_path=dump_path,
        output_dir=out_dir,
        min_freq=args.min_freq,
        max_size=args.max_size,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        x_max=args.x_max,
        alpha=args.alpha,
        lr=args.lr,
        max_epochs=args.epochs,
        limit_articles=args.limit_articles,
    )
    print(ckpt)


if __name__ == "__main__":
    main()
