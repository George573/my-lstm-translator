"""Reusable record-offset indices and sampling without replacement."""
from array import array
import csv
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import struct
import sys
import tempfile

import torch
from torch.utils.data import Dataset, Sampler

from .data import _parse_parallel_line, parallel_partition
from .model import EOS_INDEX, SOS_INDEX


class IndexedTranslationDataset(Dataset):
    """Cache byte offsets once; read and tokenize individual records on demand."""

    def __init__(self, path, source_tokenizer, target_tokenizer, source_vocabulary,
                 target_vocabulary, *, partition="train", validation_fraction=0.01,
                 test_fraction=0.01, seed=42, index_dir=None):
        if partition not in {"train", "validation", "test"}:
            raise ValueError("partition must be train, validation, or test")
        parallel_partition("", "", validation_fraction=validation_fraction,
                           test_fraction=test_fraction, seed=seed)
        self.path = Path(path).resolve()
        self.source_tokenizer, self.target_tokenizer = source_tokenizer, target_tokenizer
        self.source_vocabulary, self.target_vocabulary = source_vocabulary, target_vocabulary
        self.partition, self.seed = partition, seed
        stat = self.path.stat()
        identity = dict(version=2, partition_scheme="normalized-source-v1",
                        path=str(self.path), size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                        validation_fraction=validation_fraction,
                        test_fraction=test_fraction, seed=seed)
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = Path(index_dir) if index_dir else self.path.parent / ".lstm-index"
        cache.mkdir(parents=True, exist_ok=True)
        manifest = cache / f"{self.fingerprint}.json"
        if not manifest.exists():
            self._build_index(cache, manifest, identity)
        metadata = json.loads(manifest.read_text())
        self.count, self.columns = metadata["counts"][partition], metadata["columns"]
        self.offset_path = cache / metadata["files"][partition]
        if self.offset_path.stat().st_size != self.count * 8:
            raise ValueError("record index is incomplete; rebuild the index cache")
        self._handles = None

    def _build_index(self, cache, manifest, identity):
        counts = dict(train=0, validation=0, test=0)
        files = {}
        # Publish the manifest only after complete index files are available.
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            paths = {part: Path(temporary) / part for part in counts}
            handles = {part: path.open("wb") for part, path in paths.items()}
            try:
                with self.path.open("rb") as corpus:
                    columns = None
                    if self.path.suffix.lower() == ".csv":
                        reader = csv.reader(_decoded_lines(corpus))
                        header = next(reader, [])
                        if not {"en", "fr"}.issubset(header):
                            raise ValueError("CSV data must contain 'en' and 'fr' columns")
                        positions = {name: index for index, name in enumerate(header)}
                        columns = [positions["en"], positions["fr"]]
                    while True:
                        offset = corpus.tell()
                        if columns is not None:
                            row = next(reader, None)
                            if row is None:
                                break
                            pair = _csv_pair(row, columns)
                        else:
                            line = corpus.readline()
                            if not line:
                                break
                            pair = _tsv_pair(line)
                        if pair is None:
                            continue
                        part = parallel_partition(
                            *pair, validation_fraction=identity["validation_fraction"],
                            test_fraction=identity["test_fraction"], seed=self.seed)
                        handles[part].write(struct.pack("<Q", offset))
                        counts[part] += 1
            finally:
                for handle in handles.values():
                    handle.close()
            stat = self.path.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
                identity["size"], identity["mtime_ns"], identity["ctime_ns"]):
                raise ValueError("corpus changed while building index")
            for part, path in paths.items():
                name = f"{self.fingerprint}-{Path(temporary).name}-{part}.idx"
                path.replace(cache / name)
                files[part] = name
            temp_manifest = Path(temporary) / "manifest.json"
            temp_manifest.write_text(json.dumps(dict(counts=counts, columns=columns, files=files)))
            temp_manifest.replace(manifest)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        source, target = self.raw_pair(index)
        source_ids = self.source_vocabulary.encode(self.source_tokenizer.encode(source))
        target_ids = self.target_vocabulary.encode(self.target_tokenizer.encode(target))
        return (torch.tensor(source_ids + [EOS_INDEX], dtype=torch.long),
                torch.tensor([SOS_INDEX, *target_ids, EOS_INDEX], dtype=torch.long))

    def raw_pair(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self._handles is None or self._handles[0] != os.getpid():
            self.close()
            offsets = self.offset_path.open("rb")
            self._handles = (os.getpid(), self.path.open("rb"), offsets,
                             mmap.mmap(offsets.fileno(), 0, access=mmap.ACCESS_READ))
        _, corpus, _, offsets = self._handles
        corpus.seek(struct.unpack_from("<Q", offsets, index * 8)[0])
        if self.columns is None:
            return _tsv_pair(corpus.readline())
        return _csv_pair(next(csv.reader(_decoded_lines(corpus))), self.columns)

    def close(self):
        if getattr(self, "_handles", None) is not None:
            for handle in reversed(self._handles[1:]):
                handle.close()
            self._handles = None

    def __getstate__(self):
        return {**self.__dict__, "_handles": None}

    def __del__(self):
        self.close()


def _decoded_lines(corpus):
    for line in corpus:
        yield line.decode("utf-8")


def _tsv_pair(line):
    # Match the original text-mode parser's universal CRLF normalization.
    return _parse_parallel_line(line.decode("utf-8").replace("\r\n", "\n"))


def _csv_pair(row, columns):
    if len(row) <= max(columns):
        return None
    source, target = (row[column] for column in columns)
    return (source, target) if source and target else None


class CoverageSampler(Sampler):
    """A compact permutation; only completed optimizer updates move its cursor.

    Prefetching never changes committed state. A seed/cycle regenerates the
    permutation on resume without including a large array in each checkpoint.
    """
    def __init__(self, size, seed=42):
        if size < 1:
            raise ValueError("the training partition produced no examples")
        self.size, self.seed = size, seed
        self.cycle, self.cursor = 0, 0
        self._shuffle()

    def _shuffle(self):
        self.order = array("Q", range(self.size))
        random.Random(self.seed + self.cycle * 1_000_003).shuffle(self.order)

    def __iter__(self):
        order = self.order
        return (order[i] for i in range(self.cursor, self.size))

    def __len__(self):
        return self.size - self.cursor

    def commit(self, count):
        if count < 0 or self.cursor + count > self.size:
            raise ValueError("invalid sampler commit")
        self.cursor += count

    def next_cycle(self):
        if self.cursor != self.size:
            raise ValueError("cannot repeat rows before completing the dataset pass")
        self.cycle += 1
        self.cursor = 0
        self._shuffle()

    def state_dict(self):
        return dict(size=self.size, seed=self.seed, cycle=self.cycle, cursor=self.cursor,
                    algorithm=1, python_version=tuple(sys.version_info[:2]))

    def load_state_dict(self, state):
        if state.get("algorithm") != 1 or state.get("python_version") != tuple(sys.version_info[:2]):
            raise ValueError("resume requires the same sampler algorithm and Python version")
        if state["size"] != self.size or state["seed"] != self.seed:
            raise ValueError("sampling state does not match dataset size/seed")
        if state["cycle"] < 0 or not 0 <= state["cursor"] <= self.size:
            raise ValueError("invalid sampling state")
        self.cycle, self.cursor = state["cycle"], state["cursor"]
        self._shuffle()
