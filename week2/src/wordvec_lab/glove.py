from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from .data import build_cooccurrence, build_vocab, corpus_to_tokens


def weighting_func(x: torch.Tensor, x_max: float = 100.0, alpha: float = 0.75) -> torch.Tensor:
    return torch.where(x < x_max, (x / x_max) ** alpha, torch.ones_like(x))


class GloVeModule(LightningModule):
    def __init__(
        self,
        cooccurrence: Dict[Tuple[int, int], float],
        vocab_size: int,
        embedding_dim: int = 200,
        x_max: float = 100.0,
        alpha: float = 0.75,
        lr: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.wi = nn.Embedding(vocab_size, embedding_dim)
        self.wj = nn.Embedding(vocab_size, embedding_dim)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)
        self.lr = lr
        pairs = list(cooccurrence.items())
        self.i_idx = torch.tensor([p[0][0] for p in pairs], dtype=torch.long)
        self.j_idx = torch.tensor([p[0][1] for p in pairs], dtype=torch.long)
        self.x = torch.tensor([p[1] for p in pairs], dtype=torch.float)
        self.f = weighting_func(self.x, x_max, alpha)

    def forward(self):
        wi = self.wi(self.i_idx)
        wj = self.wj(self.j_idx)
        bi = self.bi(self.i_idx).squeeze(-1)
        bj = self.bj(self.j_idx).squeeze(-1)
        dot = torch.sum(wi * wj, dim=1)
        loss = self.f * (dot + bi + bj - torch.log(self.x + 1e-10)) ** 2
        return loss.mean()

    def training_step(self, batch, batch_idx):
        loss = self()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=self.lr)


def train_glove(
    dump_path: Path,
    output_dir: Path,
    min_freq: int = 5,
    max_size: int | None = None,
    window_size: int = 10,
    embedding_dim: int = 200,
    x_max: float = 100.0,
    alpha: float = 0.75,
    lr: float = 0.05,
    max_epochs: int = 25,
    limit_articles: int | None = None,
) -> Path:
    token_stream = list(corpus_to_tokens(dump_path, limit_articles))
    stoi, itos = build_vocab(token_stream, min_freq, max_size)
    encoded = [[stoi.get(tok, 1) for tok in seq] for seq in token_stream]
    co = build_cooccurrence(encoded, len(
        stoi), window_size=window_size, x_max=int(x_max))
    model = GloVeModule(cooccurrence=co, vocab_size=len(
        stoi), embedding_dim=embedding_dim, x_max=x_max, alpha=alpha, lr=lr)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=output_dir, filename="glove", save_top_k=1, monitor="train_loss")
    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=10,
        callbacks=[ckpt],
        default_root_dir=str(output_dir),
    )
    trainer.fit(model)
    embeddings = (model.wi.weight.data + model.wj.weight.data).cpu()
    torch.save(embeddings, output_dir / "glove.vec.pt")
    return Path(ckpt.best_model_path)
