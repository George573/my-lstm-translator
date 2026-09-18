from __future__ import annotations

import json
import heapq
import re
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from pathlib import Path
from collections.abc import Iterable, Mapping

from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Module-level worker helpers
#
# These must be defined at module level (not nested inside the class) so they
# are importable by child processes on every platform, including Windows/macOS
# which use 'spawn' as the default multiprocessing start method.
# ---------------------------------------------------------------------------

_worker_tok: BPETokenizer | None = None


def _worker_init(tok: "BPETokenizer") -> None:
    """Receive and store the tokenizer once per worker process."""
    global _worker_tok
    _worker_tok = tok


def _worker_encode(words: list[str]) -> dict[str, tuple[str, ...]]:
    """Encode a chunk of unique words inside a worker process."""
    if _worker_tok is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return {word: tuple(_worker_tok.encode_word(word)) for word in words}


def _chunk(lst: list, n: int) -> list[list]:
    """Split lst into at most n roughly-equal non-empty sub-lists."""
    k, r = divmod(len(lst), n)
    out, start = [], 0
    for i in range(n):
        end = start + k + (1 if i < r else 0)
        if start < end:
            out.append(lst[start:end])
        start = end
    return out


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """A compact byte-pair encoder with deterministic JSON persistence."""

    FORMAT_VERSION = 3

    def __init__(
        self,
        word_break: str = "</w>",
        unk_token: str = "<unk>",
        *,
        lowercase: bool = False,
        max_cache_size: int = 100_000,
        text_processing_version: int = 2,
    ):
        if max_cache_size < 0:
            raise ValueError("max_cache_size cannot be negative")
        if not word_break:
            raise ValueError("word_break cannot be empty")
        if type(text_processing_version) is not int or text_processing_version not in {1, 2}:
            raise ValueError("unsupported tokenizer text processing version")
        self.text_processing_version = text_processing_version
        self.word_break = word_break
        self.unk_token = unk_token
        self.lowercase = lowercase
        self.max_cache_size = max_cache_size
        self.rules: list[tuple[str, str]] = []
        self.tokens: set[str] = set()
        self._pair_to_rank: dict[tuple[str, str], int] = {}  # Built once, never during encode
        self._encode_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()

        # Words and punctuation are separate units. Unlike the original regex,
        # this does not silently discard punctuation.
        pattern = (r"\w+(?:['-]\w+)*'?|[^\w\s]" if text_processing_version == 2
                   else r"[^\W_]+(?:['-][^\W_]+)*'?|[^\w\s]")
        self.word_pattern = re.compile(pattern, re.UNICODE)

    @property
    def ukn_token(self) -> str:
        """Compatibility alias for tokenizer files created before v2."""
        return self.unk_token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self):
        """Build pair→rank lookup used by the heap encoder.

        Called once after training or loading — never during hot encode paths.
        O(rules) time and space.
        """
        self._pair_to_rank = {pair: rank for rank, pair in enumerate(self.rules)}

    def _extract_words(self, texts: Iterable[str]) -> Counter:
        """Extract word frequencies from a corpus efficiently."""
        words: Counter = Counter()
        for text in texts:
            words.update(self._split(text))
        return words

    def _split(self, text: str) -> list[str]:
        if self.text_processing_version == 2:
            text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.lower()
        return self.word_pattern.findall(text)

    def _remember(self, word: str, tokens: Iterable[str]) -> tuple[str, ...]:
        encoded = tuple(tokens)
        if self.max_cache_size:
            self._encode_cache[word] = encoded
            self._encode_cache.move_to_end(word)
            if len(self._encode_cache) > self.max_cache_size:
                self._encode_cache.popitem(last=False)
        return encoded

    @staticmethod
    def _merge_pair_in_tokens(tokens: list[str], pair: tuple[str, str]) -> list[str]:
        """Linear merge used during training only."""
        out, i, n = [], 0, len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                out.append(pair[0] + pair[1])
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, texts: Iterable[str], n_merges: int = 4500) -> None:
        """Train the BPE tokenizer on the provided texts."""
        if n_merges < 0:
            raise ValueError("n_merges cannot be negative")
        self._encode_cache.clear()
        print("Extracting words...")
        # vocab and word_to_tokens are local training artifacts — they are
        # discarded after the merge loop so they don't linger in memory.
        vocab: dict[str, int] = dict(self._extract_words(texts))

        # Collect the initial character set before any merges
        initial_chars: set[str] = set()
        for word in vocab:
            initial_chars.update(word)
        initial_chars.add(self.word_break)

        word_to_tokens: dict[str, list[str]] = {
            word: list(word) + [self.word_break] for word in vocab
        }

        pair_counts: Counter = Counter()
        pair_to_words: dict = defaultdict(set)

        print("Building initial pairs...")
        for word, freq in vocab.items():
            tokens = word_to_tokens[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(word)

        self.rules = []

        print(f"Training {n_merges} merges...")
        for _ in tqdm(range(n_merges)):
            if not pair_counts:
                break

            best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            self.rules.append(best_pair)

            for word in sorted(pair_to_words[best_pair]):
                freq = vocab[word]
                tokens = word_to_tokens[word]

                for i in range(len(tokens) - 1):
                    p = (tokens[i], tokens[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_to_words[p].discard(word)

                new_tokens = self._merge_pair_in_tokens(tokens, best_pair)
                word_to_tokens[word] = new_tokens

                for i in range(len(new_tokens) - 1):
                    p = (new_tokens[i], new_tokens[i + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(word)

            if best_pair in pair_counts:
                del pair_counts[best_pair]

        # Compute the final token set; vocab and word_to_tokens go out of scope here
        self.tokens = initial_chars
        for pair in self.rules:
            self.tokens.add(pair[0] + pair[1])

        self._build_index()

    # ------------------------------------------------------------------
    # Inference — single text  (O(n log n) heap-based BPE)
    # ------------------------------------------------------------------

    def encode_word(self, word: str) -> list[str]:
        """Encode a single word using a min-heap + doubly-linked list.

        Complexity: O(n log n) where n = number of characters in the word.

        Strategy
        --------
        1. Replace unknown characters with unk_token.
        2. Seed a min-heap with every adjacent pair that has a known rule,
           keyed by rule rank (lower rank = higher priority).
        3. Pop the highest-priority pair, apply the merge by updating the
           parallel prev/next arrays in O(1) (no list rebuilding), then push
           the two new neighbour pairs if they match any rule.
        4. Stale heap entries (whose pair changed due to an earlier merge) are
           detected lazily at pop time and skipped — no upfront bookkeeping.
        5. Walk the linked list once to collect the final token sequence.
        """
        cached = self._encode_cache.get(word)
        if cached is not None:
            self._encode_cache.move_to_end(word)
            return list(cached)

        # Map unknown characters to unk_token
        chars = [c if c in self.tokens else self.unk_token for c in word]
        chars.append(self.word_break)
        n = len(chars)

        if n == 1:
            return list(self._remember(word, chars))

        # Doubly-linked list over positions: O(1) merge, O(n) traversal
        #   prev[i]  = index of the previous live position  (-1 = none)
        #   next_[i] = index of the next live position      ( n = past-the-end)
        prev   = list(range(-1, n - 1))   # [-1, 0, 1, ..., n-2]
        next_  = list(range(1,  n + 1))   # [1, 2, ..., n-1, n]
        merged = [False] * n

        # Seed heap: (rule_rank, left_position) for every valid adjacent pair
        pair_to_rank = self._pair_to_rank   # local alias avoids attribute lookup in loop
        heap: list[tuple[int, int]] = []
        for i in range(n - 1):
            rank = pair_to_rank.get((chars[i], chars[i + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        rules = self.rules  # local alias

        while heap:
            rank, i = heapq.heappop(heap)

            # --- Lazily discard stale entries ---
            if merged[i]:
                continue
            j = next_[i]
            if j >= n or merged[j]:
                continue
            if chars[i] != rules[rank][0] or chars[j] != rules[rank][1]:
                continue

            # --- Apply merge: absorb j into i ---
            chars[i] += chars[j]
            merged[j] = True
            next_[i]  = next_[j]
            if next_[i] < n:
                prev[next_[i]] = i

            # Push new pairs formed with i's surviving neighbours
            pi = prev[i]
            if pi >= 0:
                rank_l = pair_to_rank.get((chars[pi], chars[i]))
                if rank_l is not None:
                    heapq.heappush(heap, (rank_l, pi))

            ni = next_[i]
            if ni < n:
                rank_r = pair_to_rank.get((chars[i], chars[ni]))
                if rank_r is not None:
                    heapq.heappush(heap, (rank_r, i))

        # Walk the linked list once to collect surviving tokens — O(n)
        result, i = [], 0
        while i < n:
            result.append(chars[i])
            i = next_[i]

        return list(self._remember(word, result))

    def encode(self, text: str) -> list[str]:
        """Encode a single text string into BPE tokens."""
        return list(chain.from_iterable(
            self.encode_word(word) for word in self._split(text)
        ))

    def decode(self, tokens: Iterable[str]) -> str:
        """Reconstruct readable text from this tokenizer's BPE tokens."""
        units: list[str] = []
        current = ""
        for token in tokens:
            if token == self.word_break:
                units.append(current)
                current = ""
            elif token.endswith(self.word_break):
                current += token[: -len(self.word_break)]
                units.append(current)
                current = ""
            else:
                current += token
        if current:
            units.append(current)

        text = " ".join(unit for unit in units if unit)
        return re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)

    # ------------------------------------------------------------------
    # Inference — batch of texts
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        texts: list[str],
        num_workers: int = 1,
    ) -> list[list[str]]:
        """Encode a list of texts into BPE token sequences.

        Parameters
        ----------
        texts       : list of raw text strings
        num_workers : worker processes to use (1 = single-process, no overhead).
                      A sensible default for large batches is os.cpu_count().
        """
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        word_lists = [self._split(text) for text in texts]
        requested_words = {word for words in word_lists for word in words}
        # Keep batch results separate from the bounded cache: even a cache of
        # size zero must encode each distinct word only once per batch.
        encoded_words = {word: self._encode_cache[word]
                         for word in requested_words if word in self._encode_cache}
        uncached = list(requested_words - encoded_words.keys())

        if uncached and num_workers > 1:
            chunks = _chunk(uncached, num_workers)
            with ProcessPoolExecutor(
                max_workers=len(chunks), initializer=_worker_init, initargs=(self,),
            ) as pool:
                for mapping in pool.map(_worker_encode, chunks):
                    encoded_words.update(mapping)
                    for word, encoded in mapping.items():
                        self._remember(word, encoded)
        else:
            encoded_words.update((word, tuple(self.encode_word(word))) for word in uncached)

        return [
            list(chain.from_iterable(encoded_words[word] for word in words))
            for words in word_lists
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the portable tokenizer state used by checkpoints and JSON."""
        return {
            "format_version": self.FORMAT_VERSION,
            "text_processing_version": self.text_processing_version,
            "word_break": self.word_break,
            "unk_token": self.unk_token,
            "lowercase": self.lowercase,
            "rules": [list(rule) for rule in self.rules],
            "tokens": sorted(self.tokens),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "BPETokenizer":
        """Restore a tokenizer, including legacy files using ``ukn_token``."""
        version = data.get("format_version", 1)
        if type(version) is not int or version not in {1, 2, cls.FORMAT_VERSION}:
            raise ValueError(f"unsupported tokenizer format version: {version!r}")
        unk_token = data.get("unk_token", data.get("ukn_token", "<unk>"))
        tokenizer = cls(
            word_break=data.get("word_break", "</w>"),
            unk_token=unk_token,
            lowercase=bool(data.get("lowercase", False)),
            text_processing_version=data["text_processing_version"] if version == 3 else 1,
        )
        tokenizer.rules = [tuple(rule) for rule in data["rules"]]
        tokenizer.tokens = set(data["tokens"])
        tokenizer._build_index()
        return tokenizer

    def save(self, filepath: str | Path) -> None:
        """Save the tokenizer to a JSON file.

        Includes the text processing version to preserve checkpoint encodings.
        """
        with Path(filepath).open("w", encoding="utf-8") as tokenizer_file:
            json.dump(self.to_dict(), tokenizer_file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "BPETokenizer":
        """Load a pre-trained tokenizer from a JSON file."""
        with Path(filepath).open(encoding="utf-8") as tokenizer_file:
            return cls.from_dict(json.load(tokenizer_file))
