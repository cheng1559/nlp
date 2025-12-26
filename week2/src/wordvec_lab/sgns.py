from __future__ import annotations

from pathlib import Path

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F

from .data import SGNSDataModule


class SkipGramNS(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lr = lr

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        center_vec = self.in_embed(centers)  # (B, D)
        pos_vec = self.out_embed(contexts)   # (B, D)
        neg_vec = self.out_embed(negatives)  # (B, K, D)
        pos_score = torch.sum(center_vec * pos_vec, dim=-1)  # (B,)
        neg_score = torch.bmm(
            neg_vec, center_vec.unsqueeze(-1)).squeeze(-1)  # (B, K)
        loss_pos = F.logsigmoid(pos_score).mean()
        loss_neg = F.logsigmoid(-neg_score).mean()
        loss = -(loss_pos + loss_neg)
        return loss

    def training_step(self, batch, batch_idx):
        centers, contexts, negs = batch
        loss = self(centers, contexts, negs)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_sgns(
    dump_path: Path,
    output_dir: Path,
    min_freq: int = 5,
    max_size: int | None = None,
    window_size: int = 5,
    negative: int = 5,
    batch_size: int = 512,
    embedding_dim: int = 200,
    lr: float = 1e-3,
    max_epochs: int = 3,
    limit_articles: int | None = None,
) -> Path:
    data = SGNSDataModule(
        dump_path=dump_path,
        min_freq=min_freq,
        max_size=max_size,
        window_size=window_size,
        negative=negative,
        batch_size=batch_size,
        limit_articles=limit_articles,
    )
    data.prepare_data()
    data.setup()
    model = SkipGramNS(vocab_size=len(data.stoi),
                       embedding_dim=embedding_dim, lr=lr)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=output_dir, filename="sgns",
                           save_top_k=1, monitor="train_loss")
    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=50,
        callbacks=[ckpt],
        default_root_dir=str(output_dir),
    )
    trainer.fit(model, datamodule=data)
    torch.save(model.in_embed.weight.data.cpu(), output_dir / "sgns.vec.pt")
    return Path(ckpt.best_model_path)
