import regex as re
from collections import defaultdict, Counter


# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#         return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# print(decode_utf8_bytes_to_str_wrong("我".encode("utf-8")))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Node:

    def __init__(self, val=None):
        self.val = val
        self.next = None

def merge_pair(word: tuple[bytes], pair: tuple[bytes]):
    out = []

    i = 0

    while i < len(word):

        if (
            i < len(word)-1
            and word[i] == pair[0]
            and word[i+1] == pair[1]
        ):
            out.append(pair[0] + pair[1])
            i += 2
        else:
            out.append(word[i])
            i += 1

    return tuple(out)
    
    
    


def train_bpe(input_path:str, vocab_size:int , special_tokens: list[str]):
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []


    with open(input_path, "r") as f:
        corpus = f.read()
    
    for byte in range(256):
        vocab[byte] = bytes([byte])

    next_token_id = len(vocab)
    
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1
        breakpoint()

    # pretokenize
    freq = defaultdict(int)
    for m in re.finditer(PAT, corpus):
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
        



if __name__ == "__main__":
    # corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest" 

    # print(train_bpe("a", 2560, []))
    breakpoint()

    