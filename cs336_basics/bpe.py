from typing import Iterable, Iterator

import regex as re
from collections import defaultdict, Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count
import time
import pickle

# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#         return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# print(decode_utf8_bytes_to_str_wrong("我".encode("utf-8")))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_pair(word: tuple[bytes], pair: tuple[bytes], new_token_id: int):
    out = []

    i = 0

    while i < len(word):

        if (
            i < len(word)-1
            and word[i] == pair[0]
            and word[i+1] == pair[1]
        ):
            out.append(new_token_id)
            i += 2
        else:
            out.append(word[i])
            i += 1

    return tuple(out)
    
    
    


def _train_bpe(input_path:str, vocab_size:int , special_tokens: list[str]):
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []


    with open(input_path, "r") as f:
        corpus = f.read()
    
    # Remove special token
    special_tokens.sort()
    special_pat = "|".join(
        re.escape(special_token)
        for special_token in special_tokens
    )

    chunks = re.split(f"({special_pat})", corpus) if special_pat else [corpus]

    
    for byte in range(256):
        vocab[byte] = bytes([byte])

    next_token_id = len(vocab)
    
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1

    # pretokenize
    freq = defaultdict(int)
    for chunk in chunks:
        if chunk not in special_tokens:
            for m in re.finditer(PAT, chunk):
                word_byte = m.group().encode("utf-8")
                freq[tuple(bytes([b]) for b in word_byte)]+=1

    # Merges
    while len(vocab) < vocab_size:
        pair_count = Counter()

        # update pair frequency
        for word_bytes, word_freq in freq.items():
            for idx in range(len(word_bytes)-1):
                adjacent_pair = (word_bytes[idx], word_bytes[idx+1])
                pair_count[adjacent_pair] += word_freq

        # find best pair
        best_pair, best_freq = max(
            pair_count.items(),
            key=lambda item: (
                item[1],  # frequency
                item[0],  # lexical order
            )
        )

        #
        new_token = best_pair[0] + best_pair[1]
        vocab[next_token_id] = new_token
        merges.append(best_pair)
        next_token_id += 1

        new_freq = {}

        for word_bytes, word_freq in freq.items():

            merged_word_bytes = merge_pair(
                word_bytes,
                best_pair
            )

            new_freq[merged_word_bytes] = word_freq
        
        freq = new_freq

    return vocab, merges

def normalize_word(word):
    if isinstance(word, int):
        return (word,)

    if isinstance(word, bytes):
        return tuple(word)

    if isinstance(word, tuple):
        if len(word) == 0:
            return word

        if isinstance(word[0], bytes):
            return tuple(b[0] for b in word)

        return tuple(word)

    return tuple(word)


def get_pairs(word):
    if len(word) < 2:
        return []

    return [
        (word[i], word[i + 1])
        for i in range(len(word) - 1)
    ]


def merge_pair_and_get_pairs(old_word, best_pair, new_token_id):
    a, b = best_pair
    n = len(old_word)

    new_word = []
    new_pairs = []

    prev_token = None
    i = 0

    while i < n:
        if (
            i + 1 < n
            and old_word[i] == a
            and old_word[i + 1] == b
        ):
            token = new_token_id
            i += 2
        else:
            token = old_word[i]
            i += 1

        if prev_token is not None:
            new_pairs.append((prev_token, token))

        new_word.append(token)
        prev_token = token

    return tuple(new_word), new_pairs


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes=1,
    file_split=1,
    stat=None,
):
    vocab: dict[int, bytes] = {}
    merges: list[tuple[int, int]] = []

    special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    for byte in range(256):
        vocab[byte] = bytes([byte])

    next_token_id = len(vocab)

    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    special_pat = "|".join(
        re.escape(token)
        for token in special_tokens
    )

    if special_pat:
        master_pat = re.compile(
            f"({special_pat})|({PAT})"
        )
    else:
        master_pat = re.compile(PAT)

    # ----------------------------------
    # Pretokenize
    # ----------------------------------
    if stat is None:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f,
                file_split,
                b"<|endoftext|>",
            )

        stat = Counter()

        tasks = [
            (
                input_path,
                master_pat,
                special_tokens,
                start_offset,
                end_offset,
            )
            for start_offset, end_offset in zip(
                boundaries[:-1],
                boundaries[1:],
            )
        ]

        with Pool(processes=num_processes) as pool:
            t0 = time.perf_counter()
            freq_stats = pool.starmap(_pretokenize, tasks)
            t1 = time.perf_counter()

        # print(f"[Pretokenize time] {t1 - t0:.2f} sec")

        # t0 = time.perf_counter()

        for freq_stat in freq_stats:
            stat.update(freq_stat)
            del freq_stat

        # t1 = time.perf_counter()

        # print(f"[Update time] {t1 - t0:.2f} sec")

        with open("stat.pickle", "wb") as f:
            pickle.dump(stat, f)
    else:
        print("stat exist")

    # ----------------------------------
    # Convert stat to word arrays
    # ----------------------------------
    words = {}
    word_freq = {}

    for word_id, (word, count) in enumerate(stat.items()):
        word = normalize_word(word)

        words[word_id] = word
        word_freq[word_id] = count

    print(len(words))
    print(len(vocab))

    # ----------------------------------
    # Build inverted index
    # ----------------------------------
    pair_count = defaultdict(int)
    pair_to_words = defaultdict(set)
    word_pairs = {}

    t0 = time.perf_counter()

    for word_id, word in words.items():
        pairs = get_pairs(word)
        word_pairs[word_id] = pairs

        count = word_freq[word_id]

        for pair in pairs:
            pair_count[pair] += count
            pair_to_words[pair].add(word_id)

    t1 = time.perf_counter()

    # print(f"[building_invert_index] {t1 - t0:.2f} sec")

    # ----------------------------------
    # Merge loop
    # ----------------------------------
    while len(vocab) < vocab_size:
        t0_loop = time.perf_counter()

        if not pair_count:
            break

        t0_max = time.perf_counter()

        best_pair, best_freq = max(
            pair_count.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
        )

        t1_max = time.perf_counter()

        # print(f"[choosing max] {t1_max - t0_max:.2f} sec")

        if best_freq <= 0:
            break

        new_token_id = next_token_id

        vocab[new_token_id] = (
            vocab[best_pair[0]]
            + vocab[best_pair[1]]
        )

        merges.append(best_pair)
        next_token_id += 1

        affected_word_ids = tuple(
            pair_to_words.pop(best_pair, ())
        )

        # print(
        #     f"best_pair={best_pair} "
        #     f"affected={len(affected_word_ids):,}"
        # )

        remove_time = 0.0
        merge_time = 0.0
        add_time = 0.0
        discard_time = 0.0
        add_set_time = 0.0

        for word_id in affected_word_ids:
            old_word = words[word_id]
            count = word_freq[word_id]
            old_pairs = word_pairs[word_id]

            # ------------------------------
            # Remove old pairs
            # ------------------------------
            t0 = time.perf_counter()

            for pair in old_pairs:
                pair_count[pair] -= count

                if pair_count[pair] <= 0:
                    del pair_count[pair]

                if pair != best_pair:
                    t_discard = time.perf_counter()

                    s = pair_to_words.get(pair)
                    if s is not None:
                        s.discard(word_id)
                        if not s:
                            pair_to_words.pop(pair, None)

                    discard_time += (
                        time.perf_counter()
                        - t_discard
                    )

            remove_time += time.perf_counter() - t0

            # ------------------------------
            # Merge word and build new pairs
            # ------------------------------
            t0 = time.perf_counter()

            new_word, new_pairs = merge_pair_and_get_pairs(
                old_word,
                best_pair,
                new_token_id,
            )

            words[word_id] = new_word
            word_pairs[word_id] = new_pairs

            merge_time += time.perf_counter() - t0

            # ------------------------------
            # Add new pairs
            # ------------------------------
            t0 = time.perf_counter()

            for pair in new_pairs:
                pair_count[pair] += count

                t_add = time.perf_counter()

                pair_to_words[pair].add(word_id)

                add_set_time += (
                    time.perf_counter()
                    - t_add
                )

            add_time += time.perf_counter() - t0

        # print(
        #     f"[Remove Pairs] {remove_time:.2f}s "
        #     f"[Merge Word] {merge_time:.2f}s "
        #     f"[Add Pairs] {add_time:.2f}s"
        # )

        # print(
        #     f"[discard()] {discard_time:.2f}s "
        #     f"[add()] {add_set_time:.2f}s"
        # )

        t1_loop = time.perf_counter()

        # print(f"[Merge] {t1_loop - t0_loop:.2f} sec")

    return vocab, merges


def _pretokenize(file_path, master_pat, special_tokens, start_offset, end_offset):
    freq = Counter()
    with open(file_path, "rb") as file: 
        file.seek(start_offset)
        chunk = file.read(end_offset - start_offset).decode("utf-8", errors="ignore")
            
        for m in master_pat.finditer(chunk):
            token = m.group()
            if token in special_tokens:
                continue
            freq[tuple(b for b in token.encode('utf-8'))] += 1

    return freq

class BPETokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.bytes_to_id = {
            v:k
            for k,v in vocab.items()
        }
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return BPETokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids = []

        if self.special_tokens:
            special_pat = "|".join(
                re.escape(token)
                for token in sorted(
                    self.special_tokens,
                    key=len,
                    reverse=True,
                )
            )

            parts = re.split(f"({special_pat})", text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            
            if self.special_tokens and part in self.special_tokens:
                ids.append(self.bytes_to_id[part.encode('utf-8')])
                continue

            for m in re.finditer(PAT, part):
                word = m.group()
                ids.extend(self.encode_word(word))
        return ids
    
    def encode_word(self, word: str):
        # 1. word -> bytes
        tokens = [bytes([b]) for b in word.encode("utf-8")]
        for a, b in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
        return [self.bytes_to_id[tok] for tok in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):           
                yield token

    def decode(self, ids: list[int]) -> str:
        return (b"".join([self.vocab[token_id] for token_id in ids])).decode("utf-8",  errors="replace")


if __name__ == "__main__":
    vocab_file = "tinystories_bpe_vocab.pkl"
    merges_file = "tinystories_bpe_merges.pkl"
    
    tokenizer = BPETokenizer.from_files(vocab_file, merges_file, ["<|endoftext|>"])
    test_string = "s"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)