from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ID_TO_LABEL, LABEL_TO_ID, CWSDataModule, encode_sequence


@dataclass
class SequenceMetrics:
    precision: float
    recall: float
    f1: float


def labels_to_tokens(chars: Sequence[str], labels: Sequence[int]) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    for ch, label_id in zip(chars, labels):
        label = ID_TO_LABEL.get(label_id, "S")
        if label == "B":
            if current:
                tokens.append("".join(current))
            current = [ch]
        elif label == "M":
            if not current:
                current = [ch]
            else:
                current.append(ch)
        elif label == "E":
            current.append(ch)
            tokens.append("".join(current))
            current = []
        else:  # "S" or fallback
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def spans_from_tokens(tokens: Sequence[str]) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    offset = 0
    for token in tokens:
        start = offset
        end = offset + len(token)
        spans.add((start, end))
        offset = end
    return spans


def sequence_prf(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> SequenceMetrics:
    pred_spans = spans_from_tokens(pred_tokens)
    gold_spans = spans_from_tokens(gold_tokens)
    tp = len(pred_spans & gold_spans)
    precision = tp / len(pred_spans) if pred_spans else 0.0
    recall = tp / len(gold_spans) if gold_spans else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall > 0 else 0.0
    return SequenceMetrics(precision, recall, f1)


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(-1)
    correct = ((preds == labels) & mask).sum().item()
    total = mask.sum().item()
    return correct / total


class BiLSTMTagger(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        lr: float = 1e-3,
        dropout: float = 0.1,
        id_to_char: Dict[int, str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(LABEL_TO_ID))
        self.lr = lr
        self.id_to_char = id_to_char

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(tokens)
        outputs, _ = self.lstm(embeds)
        outputs = self.dropout(outputs)
        return self.classifier(outputs)

    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        acc = masked_token_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        acc = masked_token_accuracy(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def segment(self, text: str, stoi: Dict[str, int]) -> List[str]:
        device = self.device if hasattr(
            self, "device") else torch.device("cpu")
        ids = encode_sequence(list(text), stoi)
        with torch.no_grad():
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            logits = self(tensor)
            preds = logits.argmax(-1).squeeze(0).tolist()
        chars = [self.id_to_char.get(
            i, "") if self.id_to_char else ch for i, ch in zip(ids, list(text))]
        return labels_to_tokens(chars, preds)


def train_model(
    data_root: Path,
    corpus: str = "pku",
    max_epochs: int = 5,
    batch_size: int = 64,
    embedding_dim: int = 128,
    hidden_size: int = 256,
    lr: float = 1e-3,
    limit: int | None = None,
    seed: int = 7,
    output_dir: Path | None = None,
) -> Path:
    seed_everything(seed, workers=True)
    data = CWSDataModule(data_root=data_root, corpus=corpus,
                         batch_size=batch_size, limit=limit)
    data.prepare_data()
    data.setup()
    model = BiLSTMTagger(
        vocab_size=data.vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        lr=lr,
        id_to_char=data.itos,
    )
    output_dir = output_dir or (data_root / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{corpus}-bilstm-{{epoch}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        callbacks=[checkpoint_cb],
    )
    trainer.fit(model, datamodule=data)
    return Path(checkpoint_cb.best_model_path)


def segment_with_model(
    checkpoint: Path,
    data_root: Path,
    corpus: str,
    text: str,
) -> List[str]:
    data = CWSDataModule(data_root=data_root, corpus=corpus,
                         batch_size=1, val_ratio=0.01)
    data.prepare_data()
    data.setup()
    model = BiLSTMTagger.load_from_checkpoint(checkpoint, id_to_char=data.itos)
    model.eval()
    return model.segment(text, data.stoi)
