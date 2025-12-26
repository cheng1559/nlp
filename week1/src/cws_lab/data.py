from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

LABEL_TO_ID = {"B": 0, "M": 1, "E": 2, "S": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def sentence_to_bmes(words: Sequence[str]) -> Tuple[List[str], List[int]]:
    chars: List[str] = []
    labels: List[int] = []
    for word in words:
        if len(word) == 0:
            continue
        if len(word) == 1:
            chars.append(word)
            labels.append(LABEL_TO_ID["S"])
        else:
            chars.extend(list(word))
            labels.extend([
                LABEL_TO_ID["B"],
                *([LABEL_TO_ID["M"]] * (len(word) - 2)),
                LABEL_TO_ID["E"],
            ])
    return chars, labels


def read_segmented_file(path: Path, limit: int | None = None) -> List[List[str]]:
    sentences: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences.append(line.split())
            if limit is not None and len(sentences) >= limit:
                break
    return sentences


def build_vocab(sentences: Iterable[Sequence[str]], min_freq: int = 1) -> Tuple[dict[str, int], dict[int, str]]:
    counter: Counter[str] = Counter()
    for sentence in sentences:
        for ch in sentence:
            counter[ch] += 1
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    stoi: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for ch, freq in sorted_items:
        if freq < min_freq:
            continue
        stoi[ch] = len(stoi)
    itos: dict[int, str] = {idx: ch for ch, idx in stoi.items()}
    return stoi, itos


def encode_sequence(sequence: Sequence[str], stoi: dict[str, int]) -> List[int]:
    unk = stoi.get(UNK_TOKEN, 1)
    return [stoi.get(ch, unk) for ch in sequence]


def collate_batch(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [len(x[0]) for x in batch]
    max_len = max(lengths)
    pad_id = 0
    pad_label = -100
    padded_tokens: List[List[int]] = []
    padded_labels: List[List[int]] = []
    for tokens, labels in batch:
        token_list = tokens.tolist()
        label_list = labels.tolist()
        token_list.extend([pad_id] * (max_len - len(token_list)))
        label_list.extend([pad_label] * (max_len - len(label_list)))
        padded_tokens.append(token_list)
        padded_labels.append(label_list)
    return torch.tensor(padded_tokens, dtype=torch.long), torch.tensor(padded_labels, dtype=torch.long)


@dataclass
class ICWB2Corpus:
    root: Path
    corpus: str = "pku"

    def training_path(self) -> Path:
        return self.root / "data" / "icwb2-data" / "training" / f"{self.corpus}_training.utf8"

    def gold_test_path(self) -> Path:
        return self.root / "data" / "icwb2-data" / "gold" / f"{self.corpus}_test_gold.utf8"

    def raw_test_path(self) -> Path:
        return self.root / "data" / "icwb2-data" / "testing" / f"{self.corpus}_test.utf8"


class CWSDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: Sequence[Tuple[List[int], List[int]]]):
        self.data = data

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, labels = self.data[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class CWSDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        corpus: str = "pku",
        batch_size: int = 64,
        min_freq: int = 1,
        val_ratio: float = 0.1,
        limit: int | None = None,
    ):
        super().__init__()
        self.corpus = corpus
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.val_ratio = val_ratio
        self.limit = limit
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def prepare_data(self):  # pragma: no cover - no external download
        corpus = ICWB2Corpus(self.data_root, self.corpus)
        if not corpus.training_path().exists():
            raise FileNotFoundError(
                f"Missing training file at {corpus.training_path()}")

    def setup(self, stage: str | None = None):
        corpus = ICWB2Corpus(self.data_root, self.corpus)
        sentences = read_segmented_file(corpus.training_path(), self.limit)
        char_sequences: List[List[str]] = []
        label_sequences: List[List[int]] = []
        for words in sentences:
            chars, labels = sentence_to_bmes(words)
            char_sequences.append(chars)
            label_sequences.append(labels)
        self.stoi, self.itos = build_vocab(char_sequences, self.min_freq)
        encoded_pairs: List[Tuple[List[int], List[int]]] = []
        for chars, labels in zip(char_sequences, label_sequences):
            encoded_pairs.append((encode_sequence(chars, self.stoi), labels))
        split = int(len(encoded_pairs) * (1 - self.val_ratio))
        train_data = encoded_pairs[:split]
        val_data = encoded_pairs[split:] if split < len(
            encoded_pairs) else encoded_pairs[-1:]
        self.train_dataset = CWSDataset(train_data)
        self.val_dataset = CWSDataset(val_data)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_batch)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_batch)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
