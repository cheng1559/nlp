"""Chinese word segmentation toolkit for week1."""

__all__ = [
    "MaximumMatcher",
    "build_dictionary",
    "ICWB2Corpus",
    "CWSDataset",
    "CWSDataModule",
    "BiLSTMTagger",
    "train_model",
    "segment_with_model",
    "segment_with_matcher",
]

__version__ = "0.1.0"

from .mmseg import MaximumMatcher, build_dictionary, segment_with_matcher
from .data import ICWB2Corpus, CWSDataset, CWSDataModule
from .model import BiLSTMTagger, train_model, segment_with_model
