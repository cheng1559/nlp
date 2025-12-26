from __future__ import annotations

import bz2
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import regex as re
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

TOKEN_RE = re.compile(r"\p{Han}+|\x17e+|[a-zA-Z]+|\d+|[\p{P}\p{S}]")
PAD = "<pad>"
UNK = "<unk>"


def clean_line(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text)
    return [tok.lower() for tok in tokens if tok.strip()]


def iter_wiki_text(dump_path: Path, limit_articles: int | None = None) -> Iterator[List[str]]:
    with bz2.open(dump_path, "rt", encoding="utf-8", errors="ignore") as f:
        buffer: List[str] = []
        articles = 0
        for line in f:
            line = line.strip()
            if line.startswith("<doc id="):
                buffer = []
                continue
            if line.startswith("</doc"):
                if buffer:
                    yield buffer
                    buffer = []
                    articles += 1
                    if limit_articles and articles >= limit_articles:
                        break
                continue
            if not line:
                continue
            buffer.append(line)


def corpus_to_tokens(dump_path: Path, limit_articles: int | None = None) -> Iterator[List[str]]:
    for article_lines in iter_wiki_text(dump_path, limit_articles):
        for line in article_lines:
            tokens = clean_line(line)
            if tokens:
                yield tokens


def build_vocab(token_iter: Iterable[Sequence[str]], min_freq: int = 5, max_size: int | None = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter: Counter[str] = Counter()
    for tokens in token_iter:
        counter.update(tokens)
    most_common = counter.most_common(max_size)
    stoi: Dict[str, int] = {PAD: 0, UNK: 1}
    for tok, freq in most_common:
        if freq < min_freq:
            continue
        stoi[tok] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def subsample(tokens: List[int], thresholds: float = 1e-4) -> List[int]:
    counts = Counter(tokens)
    total = len(tokens)
    keep: List[int] = []
    for t in tokens:
        f = counts[t] / total
        prob = 1 - math.sqrt(thresholds / f) if f > thresholds else 1.0
        if torch.rand(1).item() > prob:
            keep.append(t)
    return keep if keep else tokens


def generate_pairs(tokens: List[int], window_size: int, negative: int, vocab_size: int) -> List[Tuple[int, int, List[int]]]:
    pairs: List[Tuple[int, int, List[int]]] = []
    for i, center in enumerate(tokens):
        left = max(0, i - window_size)
        right = min(len(tokens), i + window_size + 1)
        for j in range(left, right):
            if j == i:
                continue
            context = tokens[j]
            negs = torch.randint(2, vocab_size, (negative,),
                                 dtype=torch.long).tolist()
            pairs.append((center, context, negs))
    return pairs


class SGNSDataset(Dataset[Tuple[int, int, List[int]]]):
    def __init__(self, pairs: List[Tuple[int, int, List[int]]]):
        self.pairs = pairs

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int, List[int]]:
        return self.pairs[idx]


def sgns_collate(batch: Sequence[Tuple[int, int, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers, contexts, negs = zip(*batch)
    neg_tensor = torch.tensor(negs, dtype=torch.long)
    return torch.tensor(centers, dtype=torch.long), torch.tensor(contexts, dtype=torch.long), neg_tensor


class SGNSDataModule(LightningDataModule):
    def __init__(
        self,
        dump_path: Path,
        min_freq: int = 5,
        max_size: int | None = None,
        window_size: int = 5,
        negative: int = 5,
        batch_size: int = 512,
        limit_articles: int | None = None,
    ):
        super().__init__()
        self.dump_path = dump_path
        self.min_freq = min_freq
        self.max_size = max_size
        self.window_size = window_size
        self.negative = negative
        self.batch_size = batch_size
        self.limit_articles = limit_articles
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.dataset: SGNSDataset | None = None

    def prepare_data(self):  # pragma: no cover
        if not Path(self.dump_path).exists():
            raise FileNotFoundError(f"Missing corpus at {self.dump_path}")

    def setup(self, stage: str | None = None):
        token_stream = list(corpus_to_tokens(
            self.dump_path, self.limit_articles))
        self.stoi, self.itos = build_vocab(
            token_stream, self.min_freq, self.max_size)
        encoded_articles: List[List[int]] = []
        for tokens in token_stream:
            encoded = [self.stoi.get(tok, 1) for tok in tokens]
            encoded_articles.append(subsample(encoded))
        pairs: List[Tuple[int, int, List[int]]] = []
        for encoded in tqdm(encoded_articles, desc="pairs"):
            pairs.extend(generate_pairs(encoded, self.window_size,
                         self.negative, len(self.stoi)))
        self.dataset = SGNSDataset(pairs)

    def train_dataloader(self) -> DataLoader:
        assert self.dataset is not None
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=sgns_collate)


# GloVe utilities

def build_cooccurrence(
    token_iter: Iterable[Sequence[int]],
    vocab_size: int,
    window_size: int = 10,
    x_max: int = 100,
) -> Dict[Tuple[int, int], float]:
    co = defaultdict(float)
    for tokens in token_iter:
        for i, w in enumerate(tokens):
            left = max(0, i - window_size)
            right = min(len(tokens), i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                c = tokens[j]
                distance = abs(i - j)
                increment = 1.0 / distance
                co[(w, c)] += increment
    return co
