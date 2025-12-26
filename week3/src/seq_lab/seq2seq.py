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

import json

SPECIALS = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}


def load_parallel(json_path: Path, limit: int | None = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for i, item in enumerate(data):
        pairs.append((item["source"], item["translation"]))
        if limit and i + 1 >= limit:
            break
    return pairs


def build_vocab(pairs: List[Tuple[str, str]], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter: Dict[str, int] = {}
    for src, tgt in pairs:
        for ch in list(src) + list(tgt):
            counter[ch] = counter.get(ch, 0) + 1
    stoi = dict(SPECIALS)
    for ch, freq in counter.items():
        if freq < min_freq:
            continue
        if ch not in stoi:
            stoi[ch] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    unk = stoi.get("<unk>", 3)
    return [stoi.get(ch, unk) for ch in text]


def pad_sequences(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded


class SeqDataset(Dataset[Tuple[List[int], List[int]]]):
    def __init__(self, items: List[Tuple[List[int], List[int]]]):
        self.items = items

    def __len__(self):  # pragma: no cover
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.items[idx]


def collate_batch(batch: List[Tuple[List[int], List[int]]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    src, tgt = zip(*batch)
    return pad_sequences(list(src), pad_id), pad_sequences(list(tgt), pad_id)


class Seq2SeqDataModule(LightningDataModule):
    def __init__(
        self,
        json_path: Path,
        batch_size: int = 64,
        min_freq: int = 1,
        val_ratio: float = 0.1,
        limit: int | None = None,
    ):
        super().__init__()
        self.json_path = json_path
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.val_ratio = val_ratio
        self.limit = limit
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.train_data: SeqDataset | None = None
        self.val_data: SeqDataset | None = None

    def prepare_data(self):  # pragma: no cover
        if not self.json_path.exists():
            raise FileNotFoundError(self.json_path)

    def setup(self, stage: str | None = None):
        pairs = load_parallel(self.json_path, self.limit)
        self.stoi, self.itos = build_vocab(pairs, self.min_freq)
        pad_id = self.stoi["<pad>"]
        bos = self.stoi["<s>"]
        eos = self.stoi["</s>"]
        encoded: List[Tuple[List[int], List[int]]] = []
        for src, tgt in pairs:
            src_ids = encode(src, self.stoi)
            tgt_ids = [bos] + encode(tgt, self.stoi) + [eos]
            encoded.append((src_ids, tgt_ids))
        split = int(len(encoded) * (1 - self.val_ratio))
        train_pairs = encoded[:split]
        val_pairs = encoded[split:] if split < len(encoded) else encoded[-1:]
        self.train_data = SeqDataset(train_pairs)
        self.val_data = SeqDataset(val_pairs)
        self.pad_id = pad_id

    def train_dataloader(self) -> DataLoader:
        assert self.train_data is not None
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, self.pad_id))

    def val_dataloader(self) -> DataLoader:
        assert self.val_data is not None
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, self.pad_id))


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, rnn: str):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn]
        self.rnn = rnn_cls(embed_dim, hidden, batch_first=True)

    def forward(self, src: torch.Tensor):
        emb = self.embed(src)
        outputs, state = self.rnn(emb)
        return outputs, state


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, rnn: str):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn]
        self.rnn = rnn_cls(embed_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, vocab_size)
        self.rnn_type = rnn

    def forward(self, input_tokens: torch.Tensor, state):
        emb = self.embed(input_tokens)
        output, state = self.rnn(emb, state)
        logits = self.proj(output)
        return logits, state


class Seq2SeqModule(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden: int = 512,
        rnn: str = "gru",
        lr: float = 1e-3,
        pad_id: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(vocab_size, embed_dim, hidden, rnn)
        self.decoder = Decoder(vocab_size, embed_dim, hidden, rnn)
        self.pad_id = pad_id
        self.lr = lr

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing: float = 0.5):
        enc_outputs, enc_state = self.encoder(src)
        batch, tgt_len = tgt.size()
        inputs = tgt[:, 0].unsqueeze(1)  # <s>
        logits_all: List[torch.Tensor] = []
        state = enc_state
        for t in range(1, tgt_len):
            logits, state = self.decoder(inputs, state)
            logits_all.append(logits)
            if self.training and torch.rand(1).item() < teacher_forcing:
                next_inp = tgt[:, t].unsqueeze(1)
            else:
                next_inp = logits.argmax(-1)
            inputs = next_inp
        return torch.cat(logits_all, dim=1)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src, tgt, teacher_forcing=0.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               tgt[:, 1:].reshape(-1), ignore_index=self.pad_id)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src, tgt, teacher_forcing=0.0)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               tgt[:, 1:].reshape(-1), ignore_index=self.pad_id)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_seq2seq(
    json_path: Path,
    output_dir: Path,
    embed_dim: int = 256,
    hidden: int = 512,
    rnn: str = "gru",
    batch_size: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 10,
    limit: int | None = None,
) -> Path:
    data = Seq2SeqDataModule(
        json_path=json_path, batch_size=batch_size, limit=limit)
    data.prepare_data()
    data.setup()
    model = Seq2SeqModule(vocab_size=len(data.stoi), embed_dim=embed_dim,
                          hidden=hidden, rnn=rnn, lr=lr, pad_id=data.stoi["<pad>"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=output_dir, filename=f"seq2seq-{rnn}", monitor="val_loss", save_top_k=1)
    trainer = Trainer(max_epochs=max_epochs, log_every_n_steps=50, callbacks=[
                      ckpt], default_root_dir=str(output_dir))
    trainer.fit(model, datamodule=data)
    torch.save(model.state_dict(), output_dir / f"seq2seq-{rnn}.pt")
    return Path(ckpt.best_model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train seq2seq for Tianjin->Mandarin")
    parser.add_argument("--data-root", type=str, default=str(Path(__file__).resolve(
    ).parents[2]), help="Project root containing data/Tianjin_dataset")
    parser.add_argument(
        "--corpus", type=str, default="data/Tianjin_dataset/俗世奇人part1.json", help="Parallel json path")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--rnn", type=str, default="gru",
                        choices=["rnn", "gru", "lstm"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional example cap")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    json_path = root / args.corpus
    out_dir = Path(args.output_dir) if args.output_dir else (root / "output")
    ckpt = train_seq2seq(
        json_path=json_path,
        output_dir=out_dir,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        rnn=args.rnn,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        limit=args.limit,
    )
    print(ckpt)


if __name__ == "__main__":
    main()
