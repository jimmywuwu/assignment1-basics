from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import time

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=10000,
    special_tokens=["<|endoftext|>"]
)

t0 = time.perf_counter()
tokenizer.train(
    ["data/owt_train.txt"],
    trainer
)
t1 = time.perf_counter()

print(f"Benchmark {t1 - t0:.2f} sec")