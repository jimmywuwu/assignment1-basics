import regex as re
from collections import defaultdict, Counter
from pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count
import time

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

def get_pairs(word: tuple[int, ...]):
    return zip(word, word[1:])

def train_bpe(input_path:str, vocab_size:int , special_tokens: list[str], num_processes = 1):
    vocab: dict[int, bytes] = {}
    merges:  list[tuple[int, int]] = []


    # Remove special token
    special_tokens.sort()
    special_pat = "|".join(
        re.escape(special_token)
        for special_token in special_tokens
    )
    
    for byte in range(256):
        vocab[byte] = bytes([byte])

    next_token_id = len(vocab)
    
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1
    
    special_pat = "|".join(
        re.escape(token)
        for token in sorted(
            special_tokens,
            key=len,
            reverse=True,
        )
    )

    if special_pat:
        master_pat = re.compile(
            f"({special_pat})|({PAT})"
        )
    else:
        master_pat = re.compile(PAT)

    # pretokenize

    num_processes = 15

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    stat = Counter()
    t0 = time.perf_counter()
    tasks = [(input_path, master_pat, special_tokens, start_offset, end_offset) for start_offset, end_offset in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=cpu_count()) as pool:
        t0 = time.perf_counter()
        freq_stats = pool.starmap(_pretokenize, tasks)
        t1 = time.perf_counter()

    print(f"[Pretokenize time] {t1 - t0:.2f} sec")
    
    t0 = time.perf_counter()
    for freq_stat in freq_stats:
        stat.update(freq_stat)
    
    t1 = time.perf_counter()
    print(f"[Update time] {t1 - t0:.2f} sec")

    words = {}
    word_freq = {}

    
    for word_id, (word, count) in enumerate(stat.items()):
        words[word_id] = word
        word_freq[word_id] = count

    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    pair_count = Counter()
    
    for word_id, word_bytes in words.items():
        freq = word_freq[word_id]

        for pair in zip(word_bytes, word_bytes[1:]):
            pair_count[pair] += freq
            pair_to_words[pair].add(word_id)

    while len(vocab) < vocab_size:
        t0 = time.perf_counter()

        if not pair_count:
            break

        best_pair, best_freq = max(
            pair_count.items(),
            key=lambda item: (
                item[1],
                item[0],
            )
        )

        if best_freq <= 0:
            break

        new_token_id = next_token_id
        vocab[new_token_id] = (
            vocab[best_pair[0]]
            + vocab[best_pair[1]]
        )
        merges.append(best_pair)
        next_token_id += 1

        affected_word_ids = list(pair_to_words[best_pair])

        for word_id in affected_word_ids:
            old_word = words[word_id]
            count = word_freq[word_id]

            # remove old pairs
            for pair in get_pairs(old_word):
                pair_count[pair] -= count
                if pair_count[pair] <= 0:
                    del pair_count[pair]

                pair_to_words[pair].discard(word_id)

            # merge word
            new_word = merge_pair(
                old_word,
                best_pair,
                new_token_id,
            )

            words[word_id] = new_word

            # add new pairs
            for pair in get_pairs(new_word):
                pair_count[pair] += count
                pair_to_words[pair].add(word_id)
        
        t1 = time.perf_counter()
        print(f"[Merge] {t1 - t0:.2f} sec")
        
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

# class BPETokenizer:

#     def __init__(self, vocab, merges, special_tokens=None):
#         self.vocab = vocab
#         self.merges = merges
#         self.special_tokens = special_tokens

#     def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
#         # vocab =
#         # merges =  
#         return BPETokenizer(vocab, merges, sprcial_token)

#     def encode(self, text: str) -> list[int]:
#         pass
#     def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
#         pass
#     def decode(self, ids: list[int]) -> str:
#         pass

if __name__ == "__main__":
    # corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest" 

    # print(train_bpe("a", 2560, []))

    pass    