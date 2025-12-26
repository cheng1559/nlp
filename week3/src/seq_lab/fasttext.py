from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import read_lines, simple_tokenize


def generate_ngrams(word: str, min_n: int = 3, max_n: int = 6) -> List[str]:
    extended = f"<{word}>"
    ngrams: List[str] = []
    for n in range(min_n, max_n + 1):
        ngrams.extend(extended[i: i + n]
                      for i in range(0, len(extended) - n + 1))
    return ngrams or [extended]


def build_vocab(corpus: Iterable[List[str]], min_freq: int = 5, min_n: int = 3, max_n: int = 6) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter: Counter[str] = Counter()
    for words in corpus:
        for word in words:
            counter[word] += 1
            for ng in generate_ngrams(word, min_n, max_n):
                counter[f"<ng>{ng}"] += 1
    stoi: Dict[str, int] = {"<pad>": 0}
    for token, freq in counter.items():
        if freq < min_freq:
            continue
        stoi[token] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def encode_example(words: List[str], stoi: Dict[str, int], min_n: int, max_n: int) -> List[Tuple[int, List[int]]]:
    unk = stoi.get("<unk>", None)
    encoded: List[Tuple[int, List[int]]] = []
    for w in words:
        idx = stoi.get(w, unk)
        sub = [stoi.get(f"<ng>{g}", unk) for g in generate_ngrams(
            w, min_n, max_n) if f"<ng>{g}" in stoi]
        sub = [s for s in sub if s is not None]
        if idx is None and not sub:
            continue
        encoded.append((idx if idx is not None else 0, sub))
    return encoded


class FastTextDataset(Dataset[Tuple[int, int, List[int]]]):
    def __init__(self, pairs: List[Tuple[int, int, List[int]]]):
        self.pairs = pairs

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int, List[int]]:
        return self.pairs[idx]


def fasttext_collate(batch: Sequence[Tuple[int, int, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    centers, contexts, subs = zip(*batch)
    return torch.tensor(centers, dtype=torch.long), torch.tensor(contexts, dtype=torch.long), list(subs)


class FastTextDataModule(LightningDataModule):
    def __init__(
        self,
        corpus_path: Path,
        min_freq: int = 5,
        window_size: int = 5,
        negative: int = 5,
        batch_size: int = 512,
        min_n: int = 3,
        max_n: int = 6,
        limit: int | None = None,
    ):
        super().__init__()
        self.corpus_path = corpus_path
        self.min_freq = min_freq
        self.window_size = window_size
        self.negative = negative
        self.batch_size = batch_size
        self.min_n = min_n
        self.max_n = max_n
        self.limit = limit
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.dataset: FastTextDataset | None = None

    def prepare_data(self):  # pragma: no cover
        if not self.corpus_path.exists():
            raise FileNotFoundError(self.corpus_path)

    def setup(self, stage: str | None = None):
        tokenized: List[List[str]] = []
        for i, line in enumerate(read_lines(self.corpus_path)):
            if self.limit and i >= self.limit:
                break
            tokenized.append(simple_tokenize(line))
        self.stoi, self.itos = build_vocab(
            tokenized, self.min_freq, self.min_n, self.max_n)
        pairs: List[Tuple[int, int, List[int]]] = []
        for words in tokenized:
            encoded_words = encode_example(
                words, self.stoi, self.min_n, self.max_n)
            for i, center in enumerate(encoded_words):
                c_idx, c_sub = center
                left = max(0, i - self.window_size)
                right = min(len(encoded_words), i + self.window_size + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    context_idx, _ = encoded_words[j]
                    negs = torch.randint(
                        1, len(self.stoi), (self.negative,), dtype=torch.long).tolist()
                    pairs.append((c_idx, context_idx, c_sub + negs))
        self.dataset = FastTextDataset(pairs)

    def train_dataloader(self) -> DataLoader:
        assert self.dataset is not None
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=fasttext_collate)


class FastTextModule(LightningModule):
    def __init__(self, vocab_size: int, embedding_dim: int = 200, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lr = lr

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor, subwords: List[List[int]]):
        center_vec = self.in_embed(centers)
        if any(subwords):
            max_len = max(len(s) for s in subwords)
            padded = torch.zeros(len(subwords), max_len,
                                 dtype=torch.long, device=self.device)
            for i, s in enumerate(subwords):
                if not s:
                    continue
                padded[i, : len(s)] = torch.tensor(
                    s, dtype=torch.long, device=self.device)
            mask = padded != 0
            sub_vecs = self.in_embed(padded)
            sub_mean = (sub_vecs * mask.unsqueeze(-1)).sum(dim=1) / \
                mask.sum(dim=1, keepdim=True).clamp(min=1)
            center_vec = center_vec + sub_mean
        pos_vec = self.out_embed(contexts)
        pos_score = torch.sum(center_vec * pos_vec, dim=-1)
        loss = -F.logsigmoid(pos_score).mean()
        return loss

    def training_step(self, batch, batch_idx):
        centers, contexts, subs = batch
        loss = self(centers, contexts, subs)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_fasttext(
    corpus_path: Path,
    output_dir: Path,
    embedding_dim: int = 200,
    min_freq: int = 5,
    window_size: int = 5,
    negative: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    max_epochs: int = 3,
    limit: int | None = None,
) -> Path:
    data = FastTextDataModule(
        corpus_path=corpus_path,
        min_freq=min_freq,
        window_size=window_size,
        negative=negative,
        batch_size=batch_size,
        limit=limit,
    )
    data.prepare_data()
    data.setup()
    model = FastTextModule(vocab_size=len(data.stoi),
                           embedding_dim=embedding_dim, lr=lr)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=output_dir, filename="fasttext", monitor="train_loss", save_top_k=1)
    trainer = Trainer(max_epochs=max_epochs, log_every_n_steps=50, callbacks=[
                      ckpt], default_root_dir=str(output_dir))
    trainer.fit(model, datamodule=data)
    torch.save(model.in_embed.weight.data.cpu(),
               output_dir / "fasttext.vec.pt")
    return Path(ckpt.best_model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FastText-like skip-gram")
    parser.add_argument("--data-root", type=str, default=str(
        Path(__file__).resolve().parents[2]), help="Project root containing data")
    parser.add_argument("--corpus", type=str,
                        default="data/result-rmrb.txt", help="Text corpus path")
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional line cap for quick runs")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    corpus_path = root / args.corpus
    out_dir = Path(args.output_dir) if args.output_dir else (root / "output")
    ckpt = train_fasttext(
        corpus_path=corpus_path,
        output_dir=out_dir,
        embedding_dim=args.embedding_dim,
        min_freq=args.min_freq,
        window_size=args.window_size,
        negative=args.negative,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        limit=args.limit,
    )
    print(ckpt)


if __name__ == "__main__":
    main()
