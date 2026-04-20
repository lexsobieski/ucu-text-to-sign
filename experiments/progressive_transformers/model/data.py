# coding: utf-8
"""
Data module — rewritten for modern PyTorch (no torchtext dependency).

Provides the same interface as the original:
  load_data(cfg) -> (train_data, dev_data, test_data, src_vocab, trg_vocab)
  make_data_iter(dataset, ...) -> iterable of batch objects
"""
import sys
import os
import io
from types import SimpleNamespace
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, TARGET_PAD
from vocabulary import build_vocab, Vocabulary


class SignProdDataset(TorchDataset):
    """Dataset for sign language production (gloss/text -> skeleton)."""

    def __init__(self, path, exts, fields, trg_size, skip_frames=1,
                 filter_pred=None):
        """
        Arguments:
            path: Common prefix of paths to the data files.
            exts: Tuple of extensions (.src, .trg, .files).
            fields: Tuple of (src_field, trg_field, files_field) — unused,
                    kept for API compatibility.
            trg_size: Number of values per frame (joints + counter).
            skip_frames: Skip every Nth frame.
            filter_pred: Optional predicate to filter examples.
        """
        super().__init__()

        src_path, trg_path, file_path = tuple(
            os.path.expanduser(path + x) for x in exts
        )

        self.examples = []

        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
             io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
             io.open(file_path, mode='r', encoding='utf-8') as files_file:

            for src_line, trg_line, files_line in zip(
                src_file, trg_file, files_file
            ):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                files_line = files_line.strip()

                # Tokenise source
                src_tokens = src_line.split()
                # Append EOS
                src_tokens.append(EOS_TOKEN)

                # Parse target skeleton values
                trg_values = trg_line.split(" ")
                if len(trg_values) <= 1:
                    continue
                trg_values = [float(v) + 1e-8 for v in trg_values]
                # Split into frames (skip_frames support)
                trg_frames = [
                    trg_values[i:i + trg_size]
                    for i in range(0, len(trg_values), trg_size * skip_frames)
                ]

                if not src_tokens or not trg_frames:
                    continue

                ex = SimpleNamespace(
                    src=src_tokens,
                    trg=trg_frames,
                    file_paths=files_line,
                )

                if filter_pred is not None and not filter_pred(ex):
                    continue

                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_data(cfg: dict):
    """
    Load train, dev and optionally test data as specified in configuration.

    Returns:
        train_data, dev_data, test_data, src_vocab, trg_vocab
    """
    data_cfg = cfg["data"]
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    files_lang = data_cfg.get("files", "files")
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]

    max_sent_length = data_cfg["max_sent_length"]
    # Target size is plus one due to the counter
    trg_size = cfg["model"]["trg_size"] + 1
    skip_frames = data_cfg.get("skip_frames", 1)

    exts = ("." + src_lang, "." + trg_lang, "." + files_lang)

    # Create training data with length filter
    train_data = SignProdDataset(
        path=train_path, exts=exts, fields=None,
        trg_size=trg_size, skip_frames=skip_frames,
        filter_pred=lambda x: (len(x.src) <= max_sent_length
                               and len(x.trg) <= max_sent_length),
    )

    # Build source vocabulary
    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = build_vocab(
        field="src", min_freq=src_min_freq, max_size=src_max_size,
        dataset=train_data, vocab_file=src_vocab_file,
    )

    # Target vocab is just a sized list (len = trg_size)
    trg_vocab = [None] * trg_size

    # Dev and test data (no filter)
    dev_data = SignProdDataset(
        path=dev_path, exts=exts, fields=None,
        trg_size=trg_size, skip_frames=skip_frames,
    )
    test_data = SignProdDataset(
        path=test_path, exts=exts, fields=None,
        trg_size=trg_size, skip_frames=skip_frames,
    )

    # Store vocab and trg_size on datasets for downstream use
    for ds in (train_data, dev_data, test_data):
        ds.src_vocab = src_vocab
        ds.trg_size = trg_size

    return train_data, dev_data, test_data, src_vocab, trg_vocab


def _collate_fn(batch, src_vocab, trg_size):
    """Custom collate function that produces batch objects compatible with Batch class."""
    pad_idx = src_vocab.stoi[PAD_TOKEN]

    # Numericalise source tokens
    src_seqs = []
    src_lengths = []
    trg_seqs = []
    file_paths = []

    for ex in batch:
        # Convert source tokens to indices (stoi is a defaultdict returning 0 for UNK)
        src_ids = [src_vocab.stoi[t] for t in ex.src]
        src_seqs.append(torch.tensor(src_ids, dtype=torch.long))
        src_lengths.append(len(src_ids))

        # Convert target frames to tensor
        trg_tensor = torch.tensor(ex.trg, dtype=torch.float32)
        trg_seqs.append(trg_tensor)

        file_paths.append(ex.file_paths)

    # Pad source sequences
    src_padded = pad_sequence(src_seqs, batch_first=True,
                              padding_value=pad_idx)
    src_lengths = torch.tensor(src_lengths, dtype=torch.long)

    # Pad target sequences
    max_trg_len = max(t.shape[0] for t in trg_seqs)
    trg_padded = torch.full(
        (len(trg_seqs), max_trg_len, trg_size),
        TARGET_PAD, dtype=torch.float32,
    )
    for i, t in enumerate(trg_seqs):
        trg_padded[i, :t.shape[0], :] = t

    # Return a namespace that looks like a torchtext batch
    return SimpleNamespace(
        src=(src_padded, src_lengths),
        trg=trg_padded,
        file_paths=file_paths,
    )


def make_data_iter(dataset, batch_size: int, batch_type: str = "sentence",
                   train: bool = False, shuffle: bool = False):
    """
    Returns a DataLoader for the dataset.

    Maintains the same call signature as the original torchtext-based version.
    """
    src_vocab = dataset.src_vocab
    trg_size = getattr(dataset, 'trg_size',
                       len(dataset[0].trg[0]) if len(dataset) > 0 else 151)

    def collate(batch):
        return _collate_fn(batch, src_vocab, trg_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if train else False,
        collate_fn=collate,
        drop_last=False,
    )
