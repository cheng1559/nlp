"""Sequence lab: FastText-like embeddings, seq2seq translation, char-level tagging."""

__all__ = [
    "FastTextModule",
    "FastTextDataModule",
    "Seq2SeqModule",
    "Seq2SeqDataModule",
    "CharTagger",
    "TagDataModule",
]

from .fasttext import FastTextModule, FastTextDataModule
from .seq2seq import Seq2SeqModule, Seq2SeqDataModule
from .tagging import CharTagger, TagDataModule
