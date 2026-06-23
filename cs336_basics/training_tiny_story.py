import time
import pickle
import tracemalloc
from bpe import train_bpe

start = time.time()
tracemalloc.start()

vocab, merges = train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

elapsed = time.time() - start
peak_gb = peak / 1024**3

with open("tinystories_bpe_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("tinystories_bpe_merges.pkl", "wb") as f:
    pickle.dump(merges, f)

longest_id, longest_token = max(
    vocab.items(),
    key=lambda x: len(x[1])
)

print("time:", elapsed)
print("peak GB:", peak_gb)
print("longest token id:", longest_id)
print("longest token bytes:", longest_token)
print("longest token decoded:", longest_token.decode("utf-8", errors="replace"))
print("longest token length:", len(longest_token))

breakpoint()