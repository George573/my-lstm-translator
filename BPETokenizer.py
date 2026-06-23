import os
import re
import json
import heapq
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Module-level worker helpers
#
# These must be defined at module level (not nested inside the class) so they
# are importable by child processes on every platform, including Windows/macOS
# which use 'spawn' as the default multiprocessing start method.
# ---------------------------------------------------------------------------

_worker_tok: Optional["BPETokenizer"] = None


def _worker_init(tok: "BPETokenizer") -> None:
    """Receive and store the tokenizer once per worker process."""
    global _worker_tok
    _worker_tok = tok


def _worker_encode(words: List[str]) -> Dict[str, List[str]]:
    """Encode a chunk of unique words inside a worker process."""
    return {w: _worker_tok.encode_word(w) for w in words}


def _chunk(lst: list, n: int) -> List[list]:
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
    def __init__(self, word_break: str = '</w>', ukn_token: str = '<ukn>'):
        self.word_break = word_break
        self.ukn_token = ukn_token
        self.rules: List[Tuple[str, str]] = []
        self.tokens: Set[str] = set()
        self._pair_to_rank: Dict[Tuple[str, str], int] = {}  # Built once, never during encode
        self._encode_cache: Dict[str, List[str]] = {}        # Per-word memoisation

        # Pre-compile regex for fast C-level execution
        self.word_pattern = re.compile(r'[^\W_]+(?:[\'\-][^\W_]+)*\'?')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self):
        """Build pair→rank lookup used by the heap encoder.

        Called once after training or loading — never during hot encode paths.
        O(rules) time and space.
        """
        self._pair_to_rank = {pair: rank for rank, pair in enumerate(self.rules)}

    def _extract_words(self, texts: List[str]) -> Counter:
        """Extract word frequencies from a corpus efficiently."""
        return Counter(self.word_pattern.findall("\n".join(texts)))

    @staticmethod
    def _merge_pair_in_tokens(tokens: List[str], pair: Tuple[str, str]) -> List[str]:
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

    def train(self, texts: List[str], n_merges: int = 4500):
        """Train the BPE tokenizer on the provided texts."""
        print("Extracting words...")
        # vocab and word_to_tokens are local training artifacts — they are
        # discarded after the merge loop so they don't linger in memory.
        vocab: Dict[str, int] = dict(self._extract_words(texts))

        # Collect the initial character set before any merges
        initial_chars: Set[str] = set()
        for word in vocab:
            initial_chars.update(word)
        initial_chars.add(self.word_break)

        word_to_tokens: Dict[str, List[str]] = {
            word: list(word) + [self.word_break] for word in vocab
        }

        pair_counts: Counter = Counter()
        pair_to_words: Dict = defaultdict(set)

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

            best_pair = max(pair_counts, key=pair_counts.get)
            self.rules.append(best_pair)

            for word in list(pair_to_words[best_pair]):
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

    def encode_word(self, word: str) -> List[str]:
        """Encode a single word using a min-heap + doubly-linked list.

        Complexity: O(n log n) where n = number of characters in the word.

        Strategy
        --------
        1. Replace unknown characters with ukn_token.
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
            return cached

        # Map unknown characters to ukn_token
        chars = [c if c in self.tokens else self.ukn_token for c in word]
        chars.append(self.word_break)
        n = len(chars)

        if n == 1:
            self._encode_cache[word] = chars
            return chars

        # Doubly-linked list over positions: O(1) merge, O(n) traversal
        #   prev[i]  = index of the previous live position  (-1 = none)
        #   next_[i] = index of the next live position      ( n = past-the-end)
        prev   = list(range(-1, n - 1))   # [-1, 0, 1, ..., n-2]
        next_  = list(range(1,  n + 1))   # [1, 2, ..., n-1, n]
        merged = [False] * n

        # Seed heap: (rule_rank, left_position) for every valid adjacent pair
        pair_to_rank = self._pair_to_rank   # local alias avoids attribute lookup in loop
        heap: List[Tuple[int, int]] = []
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

        self._encode_cache[word] = result
        return result

    def encode(self, text: str) -> List[str]:
        """Encode a single text string into BPE tokens."""
        return list(chain.from_iterable(
            self.encode_word(w) for w in self.word_pattern.findall(text)
        ))

    # ------------------------------------------------------------------
    # Inference — batch of texts
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        texts: List[str],
        num_workers: int = 1,
    ) -> List[List[str]]:
        """Encode a list of texts into BPE token sequences.

        Parameters
        ----------
        texts       : list of raw text strings
        num_workers : worker processes to use (1 = single-process, no overhead).
                      A sensible default for large batches is os.cpu_count().
        """
        # Step 1: split every text into its word list — regex runs in C
        word_lists: List[List[str]] = [self.word_pattern.findall(t) for t in texts]

        # Step 2: find unique words not already in the cache
        uncached = list(
            {w for wl in word_lists for w in wl} - self._encode_cache.keys()
        )

        # Step 3: encode uncached words — single-process or parallel
        if uncached:
            if num_workers > 1:
                # Distribute words evenly; fewer chunks if words < workers
                chunks = _chunk(uncached, num_workers)
                with ProcessPoolExecutor(
                    max_workers=len(chunks),
                    initializer=_worker_init,
                    initargs=(self,),
                ) as pool:
                    for mapping in pool.map(_worker_encode, chunks):
                        self._encode_cache.update(mapping)
            else:
                for word in uncached:
                    self.encode_word(word)

        # Step 4: reassemble — pure cache lookups, zero re-encoding
        return [
            list(chain.from_iterable(self._encode_cache[w] for w in wl))
            for wl in word_lists
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Save the tokenizer to a JSON file.

        Compatible with inference implementations expecting
        (rules, tokens, ukn_token). Vocab is never written.
        """
        data = {
            'word_break': self.word_break,
            'ukn_token':  self.ukn_token,
            'rules':      self.rules,          # List[Tuple] → List[List] in JSON
            'tokens':     list(self.tokens),   # Set → List for JSON
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Load a pre-trained tokenizer from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tok = cls(word_break=data['word_break'], ukn_token=data['ukn_token'])
        tok.rules  = [tuple(r) for r in data['rules']]
        tok.tokens = set(data['tokens'])
        tok._build_index()   # O(rules) — done once here, never during encode
        return tok