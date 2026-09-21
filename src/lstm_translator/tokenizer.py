from __future__ import annotations

import json
import heapq
import logging
import re
import unicodedata
from collections import Counter, OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from itertools import chain, islice
from pathlib import Path
from collections.abc import Iterable, Mapping, Sized

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

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
    return {word: _worker_tok._encode_word_tuple(word) for word in words}


def _worker_extract(texts: list[str]) -> Counter:
    if _worker_tok is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return _worker_tok._extract_words(texts, show_progress=False)


_merge_vocab: dict[str, int] = {}
_merge_tokens: dict[str, list[str]] = {}
_merge_index: dict[tuple[str, str], set[str]] = {}


def _worker_merge_init(vocab: dict[str, int], word_break: str) -> None:
    global _merge_vocab, _merge_tokens, _merge_index
    _merge_vocab = vocab
    _merge_tokens = {word: list(word) + [word_break] for word in vocab}
    _merge_index = {}
    for word, tokens in _merge_tokens.items():
        for pair in zip(tokens, tokens[1:]):
            holders = _merge_index.get(pair)
            if holders is None:
                _merge_index[pair] = {word}
            else:
                holders.add(word)


def _worker_merge(pair: tuple[str, str] | None) -> dict[tuple[str, str], int]:
    """Return initial shard counts, or the *net* count changes after a merge.

    Most pairs of an affected word are subtracted and re-added unchanged; those
    cancel out here and are dropped, which keeps the reply (IPC) small.
    """
    delta: dict[tuple[str, str], int] = {}
    if pair is None:
        for word, tokens in _merge_tokens.items():
            freq = _merge_vocab[word]
            for adjacent in zip(tokens, tokens[1:]):
                delta[adjacent] = delta.get(adjacent, 0) + freq
        return delta

    index = _merge_index
    words = index.pop(pair, None)  # popped: safe to iterate while updating the index
    if not words:
        return delta
    left, right = pair
    merged = left + right
    for word in words:  # holder sets may be stale; the diff skips words without a match
        result = _merge_word_diff(_merge_tokens[word], left, right, merged)
        if result is None:
            continue
        tokens, removed, added = result
        _merge_tokens[word] = tokens
        freq = _merge_vocab[word]
        for adjacent in removed:
            delta[adjacent] = delta.get(adjacent, 0) - freq
        for adjacent in added:
            delta[adjacent] = delta.get(adjacent, 0) + freq
            holders = index.get(adjacent)
            if holders is None:
                index[adjacent] = {word}
            else:
                holders.add(word)
    return {adjacent: change for adjacent, change in delta.items() if change}


def _merge_word_diff(
    tokens: list[str], left: str, right: str, merged: str,
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]] | None:
    """Merge ``left right`` -> ``merged`` (greedy, left to right) and report the
    adjacent-pair occurrences that disappear and appear.

    Returns None if the pair does not occur (a stale holder entry). Only pairs
    touching a match site change: for a match at old position i these are old
    pairs i-1, i, i+1 and, in the new list, the pairs on both sides of the merged
    token. Indices are deduplicated so pairs shared by adjacent matches count once.
    """
    n = len(tokens)
    out: list[str] = []
    matches: list[int] = []      # old positions of matches
    merged_at: list[int] = []    # positions of merged tokens in `out`
    i = 0
    while i < n:
        if i < n - 1 and tokens[i] == left and tokens[i + 1] == right:
            matches.append(i)
            merged_at.append(len(out))
            out.append(merged)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    if not matches:
        return None
    removed_idx: set[int] = set()
    for i in matches:
        removed_idx.update((i - 1, i, i + 1))
    last = n - 2
    removed = [(tokens[j], tokens[j + 1]) for j in removed_idx if 0 <= j <= last]
    added_idx: set[int] = set()
    for k in merged_at:
        added_idx.update((k - 1, k))
    last_new = len(out) - 2
    added = [(out[k], out[k + 1]) for k in added_idx if 0 <= k <= last_new]
    return out, removed, added


def _pop_best_pair(heap: list, counts: dict) -> tuple[str, str] | None:
    """Pop the pair with the highest count (ties: smallest pair) from a lazy heap.

    Invariant: every live pair has a heap entry whose recorded count is >= its
    current count. Counts only *increase* for pairs containing a freshly merged
    token, and callers push those eagerly; decreases are repaired here, when a
    stale entry surfaces. Ordering is (-count, pair), i.e. exactly the key of the
    former ``min(pair_counts, key=lambda p: (-pair_counts[p], p))``.
    """
    while heap:
        negative, pair = heapq.heappop(heap)
        current = counts.get(pair)
        if current is None:
            continue  # pair no longer exists
        recorded = -negative
        if current == recorded:
            return pair
        if current < recorded:  # count dropped since this entry was pushed
            heapq.heappush(heap, (-current, pair))
        # current > recorded: a fresher entry was pushed when the count rose
    return None


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


# Below this many *uncached unique words per worker*, process start-up (~10 ms
# with fork, 100+ ms with spawn) costs more than the ~7 us/word it would save.
_MIN_WORDS_PER_WORKER = 10_000


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """A compact byte-pair encoder with deterministic JSON persistence."""

    FORMAT_VERSION = 3

    # Pre-tokenisation patterns, selected by ``text_processing_version``.
    #   1 (legacy): words and punctuation; whitespace and ``_`` are dropped.
    #   2 (legacy): as 1 but keeps ``_``; NFC-normalised. Whitespace is dropped,
    #               so ``decode`` can only approximate the original spacing.
    #   3         : lossless. The matches partition the (normalised) text: one
    #               leading space attaches to the following word/punctuation mark
    #               (`` hello``), any other whitespace run is its own unit, and
    #               nothing is dropped. ``decode`` is exact concatenation.
    _PATTERNS = {
        1: r"[^\W_]+(?:['-][^\W_]+)*'?|[^\w\s]",
        2: r"\w+(?:['-]\w+)*'?|[^\w\s]",
        3: r" ?(?:\w+(?:['-]\w+)*'?|[^\w\s])|\s+(?!\S)|\s+",
    }

    def __init__(
        self,
        word_break: str = "</w>",
        unk_token: str = "<unk>",
        *,
        lowercase: bool = False,
        max_cache_size: int = 100_000,
        text_processing_version: int = 3,
    ):
        if max_cache_size < 0:
            raise ValueError("max_cache_size cannot be negative")
        if not word_break:
            raise ValueError("word_break cannot be empty")
        if type(text_processing_version) is not int or text_processing_version not in self._PATTERNS:
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

        pattern = self._PATTERNS[text_processing_version]
        self.word_pattern = re.compile(pattern, re.UNICODE)

    def __getstate__(self) -> dict:
        # The cache is a pure accelerator and can hold ~100k entries; the rank
        # table is derived from `rules`. Neither should be pickled to workers.
        state = self.__dict__.copy()
        state["_encode_cache"] = OrderedDict()
        state["_pair_to_rank"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._build_index()

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

    def _extract_words(
        self, texts: Iterable[str], num_workers: int = 1, chunk_size: int = 2_000,
        *, show_progress: bool = True,
    ) -> Counter:
        """Extract word frequencies from a corpus efficiently."""
        words: Counter = Counter()
        total = len(texts) if isinstance(texts, Sized) else None
        iterator = iter(texts)
        with tqdm(total=total, desc="Extracting words", unit="texts",
                  disable=not show_progress) as progress:
            if num_workers > 1:
                # Send only text-processing configuration, not existing rules/cache.
                worker_tokenizer = BPETokenizer(
                    lowercase=self.lowercase,
                    text_processing_version=self.text_processing_version,
                    max_cache_size=0,
                )
                with ProcessPoolExecutor(
                    max_workers=num_workers, initializer=_worker_init,
                    initargs=(worker_tokenizer,),
                ) as pool:
                    pending = deque()
                    # Bound submitted work: Executor.map eagerly consumes generators
                    # on supported Python versions before 3.14.
                    for _ in range(2 * num_workers):
                        chunk = list(islice(iterator, chunk_size))
                        if not chunk:
                            break
                        pending.append((pool.submit(_worker_extract, chunk), len(chunk)))
                    while pending:
                        future, count = pending.popleft()
                        words.update(future.result())
                        progress.update(count)
                        chunk = list(islice(iterator, chunk_size))
                        if chunk:
                            pending.append((pool.submit(_worker_extract, chunk), len(chunk)))
                return words
            while chunk := list(islice(iterator, chunk_size)):
                for text in chunk:
                    words.update(self._split(text))
                progress.update(len(chunk))
        return words

    def _split(self, text: str) -> list[str]:
        if self.text_processing_version >= 2:
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

    def train(
        self, texts: Iterable[str], n_merges: int = 4500, *,
        num_workers: int = 1, chunk_size: int = 2_000,
    ) -> None:
        """Train BPE with optional parallel word extraction and pair counting.

        Workers are processes; merge selection and input iteration remain serial.
        At most twice ``num_workers`` chunks are queued at a time.
        """
        if n_merges < 0:
            raise ValueError("n_merges cannot be negative")
        for name, value in (("num_workers", num_workers), ("chunk_size", chunk_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._encode_cache.clear()
        # vocab and word_to_tokens are local training artifacts — they are
        # discarded after the merge loop so they don't linger in memory.
        vocab: dict[str, int] = dict(self._extract_words(texts, num_workers, chunk_size))

        # Collect the initial character set before any merges
        initial_chars: set[str] = set()
        for word in vocab:
            initial_chars.update(word)
        initial_chars.add(self.word_break)

        if num_workers > 1:
            self.rules = self._train_parallel_merges(vocab, n_merges, num_workers)
            self.tokens = initial_chars | {left + right for left, right in self.rules}
            self._build_index()
            return

        word_to_tokens: dict[str, list[str]] = {
            word: list(word) + [self.word_break] for word in vocab
        }

        # Invariants: every pair in pair_counts has a holder set that is a superset of
        # the words containing it; both are dropped together when the count hits 0.
        pair_counts: dict[tuple[str, str], int] = {}
        pair_to_words: dict[tuple[str, str], set[str]] = {}

        logger.info("Building initial pairs...")
        for word, freq in vocab.items():
            tokens = word_to_tokens[word]
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                holders = pair_to_words.get(pair)
                if holders is None:
                    pair_to_words[pair] = {word}
                else:
                    holders.add(word)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        self.rules = []

        logger.info("Training %d merges...", n_merges)
        for _ in tqdm(range(n_merges)):
            best_pair = _pop_best_pair(heap, pair_counts)
            if best_pair is None:
                break
            self.rules.append(best_pair)
            left, right = best_pair
            merged = left + right
            created: set[tuple[str, str]] = set()

            # Holder sets are supersets (a word is not un-listed when a merge
            # removes a pair from it); words without a match are skipped. Integer
            # count arithmetic is order-independent, so no sorting is needed.
            for word in pair_to_words.pop(best_pair):
                result = _merge_word_diff(word_to_tokens[word], left, right, merged)
                if result is None:
                    continue
                new_tokens, removed, added = result
                word_to_tokens[word] = new_tokens
                freq = vocab[word]

                for pair in removed:
                    remaining = pair_counts[pair] - freq
                    if remaining > 0:
                        pair_counts[pair] = remaining
                    else:
                        del pair_counts[pair]
                        pair_to_words.pop(pair, None)

                for pair in added:
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
                    holders = pair_to_words.get(pair)
                    if holders is None:
                        pair_to_words[pair] = {word}
                    else:
                        holders.add(word)
                    created.add(pair)

            # Only pairs touching the new token can have gained count.
            for pair in created:
                count = pair_counts.get(pair)
                if count is not None:
                    heapq.heappush(heap, (-count, pair))

        # Compute the final token set; vocab and word_to_tokens go out of scope here
        self.tokens = initial_chars
        for pair in self.rules:
            self.tokens.add(pair[0] + pair[1])

        self._build_index()

    def _train_parallel_merges(
        self, vocab: dict[str, int], n_merges: int, num_workers: int,
    ) -> list[tuple[str, str]]:
        if not vocab or not n_merges:
            return []
        rules: list[tuple[str, str]] = []
        logger.info("Building initial pairs in parallel...")
        with ExitStack() as stack:
            # Each single-worker pool owns one persistent shard. A shared pool
            # does not guarantee that successive tasks reach the same worker.
            pools = [stack.enter_context(ProcessPoolExecutor(
                max_workers=1, initializer=_worker_merge_init,
                initargs=({word: vocab[word] for word in shard}, self.word_break),
            )) for shard in _chunk(list(vocab), min(num_workers, len(vocab)))]
            counts: dict[tuple[str, str], int] = {}
            for future in [pool.submit(_worker_merge, None) for pool in pools]:
                for pair, count in future.result().items():
                    counts[pair] = counts.get(pair, 0) + count
            heap = [(-count, pair) for pair, count in counts.items()]
            heapq.heapify(heap)

            logger.info("Training %d merges...", n_merges)
            for _ in tqdm(range(n_merges)):
                best = _pop_best_pair(heap, counts)
                if best is None:
                    break
                rules.append(best)
                merged = best[0] + best[1]
                futures = [pool.submit(_worker_merge, best) for pool in pools]
                changed: set[tuple[str, str]] = set()
                for future in futures:
                    # Shard deltas may cancel across shards, so all of them are
                    # summed before any pair is deleted or pushed.
                    for pair, delta in future.result().items():
                        counts[pair] = counts.get(pair, 0) + delta
                        changed.add(pair)
                for pair in changed:
                    count = counts[pair]
                    if count <= 0:
                        del counts[pair]
                    elif pair[0] == merged or pair[1] == merged:
                        heapq.heappush(heap, (-count, pair))
        return rules

    # ------------------------------------------------------------------
    # Inference — single text  (O(n log n) heap-based BPE)
    # ------------------------------------------------------------------

    def encode_word(self, word: str) -> list[str]:
        """Encode a single word into BPE tokens (returns a fresh list)."""
        return list(self._encode_word_tuple(word))

    def _encode_word_tuple(self, word: str) -> tuple[str, ...]:
        """Encode a single word using a min-heap + doubly-linked list.

        Returns the (immutable, possibly cached) token tuple without copying.

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
            return cached

        # Map unknown characters to unk_token
        chars = [c if c in self.tokens else self.unk_token for c in word]
        chars.append(self.word_break)
        n = len(chars)

        if n == 1:
            return self._remember(word, chars)

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

        return self._remember(word, result)

    def encode(self, text: str) -> list[str]:
        """Encode a single text string into BPE tokens."""
        return list(chain.from_iterable(map(self._encode_word_tuple, self._split(text))))

    def decode(self, tokens: Iterable[str]) -> str:
        """Reconstruct text from this tokenizer's BPE tokens.

        With ``text_processing_version >= 3`` this is exact: ``decode(encode(s))``
        equals ``s`` after NFC normalisation (and lower-casing if ``lowercase``),
        provided every character of ``s`` was in the training alphabet (unknown
        characters become ``unk_token``). Legacy versions (1, 2) discarded
        whitespace when tokenising, so they only approximate the original spacing.
        """
        if self.text_processing_version >= 3:
            word_break, cut = self.word_break, -len(self.word_break)
            return "".join(t[:cut] if t.endswith(word_break) else t for t in tokens)
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

        # Cap the pool so every worker gets enough work to repay its start-up.
        workers = min(num_workers, len(uncached) // _MIN_WORDS_PER_WORKER)
        if workers > 1:
            chunks = _chunk(uncached, workers)
            with ProcessPoolExecutor(
                max_workers=len(chunks), initializer=_worker_init, initargs=(self,),
            ) as pool:
                for mapping in pool.map(_worker_encode, chunks):
                    encoded_words.update(mapping)
                    for word, encoded in mapping.items():
                        self._remember(word, encoded)
        else:
            encoded_words.update((word, self._encode_word_tuple(word)) for word in uncached)

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
