from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

LABEL_MAP = {
    "POS": {"N": 0, "V": 1, "ADJ": 2, "ADV": 3, "PRON": 4, "DET": 5, "NUM": 6, "PUNCT": 7, "OTHER": 8},
    "NER": {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6},
}
SPECIALS = {"<pad>": 0, "<unk>": 1}


def read_corpus(path: Path, task: str) -> List[Tuple[List[str], List[str]]]:
    pairs: List[Tuple[List[str], List[str]]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split()
            chars, labels = [], []
            for it in items:
                if "/" not in it:
                    continue
                ch, tag = it.rsplit("/", 1)
                chars.extend(list(ch))
                if task == "POS":
                    labels.extend([tag] * len(ch))
                else:
                    if tag == "O":
                        labels.extend(["O"] * len(ch))
                    else:
                        prefix = "B-" + tag
                        labels.append(prefix)
                        labels.extend(["I-" + tag] * (len(ch) - 1))
            if chars and labels and len(chars) == len(labels):
                pairs.append((chars, labels))
    return pairs


def build_vocab(seqs: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    counter: Dict[str, int] = {}
    for seq in seqs:
        for ch in seq:
            counter[ch] = counter.get(ch, 0) + 1
    stoi = dict(SPECIALS)
    for ch, freq in counter.items():
        if freq < min_freq:
            continue
        if ch not in stoi:
            stoi[ch] = len(stoi)
    return stoi


def encode(chars: List[str], stoi: Dict[str, int]) -> List[int]:
    unk = stoi.get("<unk>", 1)
    return [stoi.get(c, unk) for c in chars]


def pad_batch(tokens: List[List[int]], labels: List[List[int]], pad_id: int, label_pad: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(t) for t in tokens)
    tok_tensor = torch.full((len(tokens), max_len), pad_id, dtype=torch.long)
    lab_tensor = torch.full((len(tokens), max_len),
                            label_pad, dtype=torch.long)
    for i, (t, l) in enumerate(zip(tokens, labels)):
        tok_tensor[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        lab_tensor[i, : len(l)] = torch.tensor(l, dtype=torch.long)
    return tok_tensor, lab_tensor


class TagDataset(Dataset[Tuple[List[int], List[int]]]):
    def __init__(self, items: List[Tuple[List[int], List[int]]]):
        self.items = items

    def __len__(self):  # pragma: no cover
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.items[idx]


def collate_fn(batch: List[Tuple[List[int], List[int]]], pad_id: int, label_pad: int):
    tokens, labels = zip(*batch)
    return pad_batch(list(tokens), list(labels), pad_id, label_pad)


class TagDataModule(LightningDataModule):
    def __init__(self, corpus: Path, task: str, batch_size: int = 64, min_freq: int = 1, val_ratio: float = 0.1):
        super().__init__()
        self.corpus = corpus
        self.task = task
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.val_ratio = val_ratio
        self.stoi: Dict[str, int] = {}
        self.label_map = LABEL_MAP[task]
        self.label_pad = len(self.label_map)

    def prepare_data(self):  # pragma: no cover
        if not self.corpus.exists():
            raise FileNotFoundError(self.corpus)

    def setup(self, stage: str | None = None):
        pairs = read_corpus(self.corpus, self.task)
        self.stoi = build_vocab([p[0] for p in pairs], self.min_freq)
        encoded: List[Tuple[List[int], List[int]]] = []
        for chars, labels in pairs:
            tok_ids = encode(chars, self.stoi)
            lab_ids = [self.label_map.get(l, self.label_pad) for l in labels]
            encoded.append((tok_ids, lab_ids))
        split = int(len(encoded) * (1 - self.val_ratio))
        train_items = encoded[:split]
        val_items = encoded[split:] if split < len(encoded) else encoded[-1:]
        self.train_ds = TagDataset(train_items)
        self.val_ds = TagDataset(val_items)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, self.stoi["<pad>"], self.label_pad))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, self.stoi["<pad>"], self.label_pad))


class CharTagger(LightningModule):
    def __init__(self, vocab_size: int, num_labels: int, embed_dim: int = 128, hidden: int = 256, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_labels)
        self.lr = lr
        self.label_pad = num_labels

    def forward(self, tokens: torch.Tensor):
        emb = self.embed(tokens)
        outputs, _ = self.encoder(emb)
        logits = self.fc(outputs)
        return logits

    def step(self, batch, stage: str):
        tokens, labels = batch
        logits = self(tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=self.label_pad)
        pred = logits.argmax(-1)
        mask = labels != self.label_pad
        correct = ((pred == labels) & mask).sum().item()
        total = mask.sum().item()
        acc = correct / total if total else 0.0
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_tagger(
    corpus: Path,
    task: str,
    output_dir: Path,
    embed_dim: int = 128,
    hidden: int = 256,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 5,
):
    dm = TagDataModule(corpus=corpus, task=task, batch_size=batch_size)
    dm.prepare_data()
    dm.setup()
    model = CharTagger(vocab_size=len(dm.stoi), num_labels=len(
        dm.label_map) + 1, embed_dim=embed_dim, hidden=hidden, lr=lr)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=output_dir, filename=f"tagger-{task.lower()}", monitor="val_acc", mode="max", save_top_k=1)
    trainer = Trainer(max_epochs=epochs, log_every_n_steps=50, callbacks=[
                      ckpt], default_root_dir=str(output_dir))
    trainer.fit(model, datamodule=dm)
    torch.save(model.state_dict(), output_dir / f"tagger-{task.lower()}.pt")
    return Path(ckpt.best_model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Char-level POS/NER tagger")
    parser.add_argument("--data-root", type=str, default=str(Path(__file__).resolve(
    ).parents[2]), help="Project root containing data/result-rmrb.txt")
    parser.add_argument("--corpus", type=str, default="data/result-rmrb.txt")
    parser.add_argument("--task", type=str, default="POS",
                        choices=["POS", "NER"])
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    corpus = root / args.corpus
    out_dir = Path(args.output_dir) if args.output_dir else (root / "output")
    ckpt = train_tagger(
        corpus=corpus,
        task=args.task,
        output_dir=out_dir,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
    )
    print(ckpt)


if __name__ == "__main__":
    main()
