# prepare_tinystories_hf_bpe.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKEN = "<|endoftext|>"


def iter_texts(dataset, text_column: str):
    for example in dataset:
        text = example[text_column]
        if text is None:
            continue
        yield text


def train_bpe_tokenizer(
    train_dataset,
    vocab_size: int,
    text_column: str,
    output_dir: Path,
) -> Tokenizer:
    tokenizer = Tokenizer(
        BPE(
            unk_token=None,
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[SPECIAL_TOKEN],
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        iter_texts(train_dataset, text_column),
        trainer=trainer,
    )

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    tokenizer.model.save(
        str(output_dir),
        "hf_bpe",
    )

    print(f"Saved tokenizer to {tokenizer_path}")
    print(f"Saved vocab/merges to {output_dir}")

    return tokenizer


def encode_dataset_to_bin(
    dataset,
    tokenizer: Tokenizer,
    output_path: Path,
    text_column: str,
    dtype=np.uint16,
):
    all_ids: list[int] = []

    end_token_id = tokenizer.token_to_id(SPECIAL_TOKEN)

    if end_token_id is None:
        raise ValueError(
            f"Tokenizer does not contain special token {SPECIAL_TOKEN}"
        )

    for example in dataset:
        text = example[text_column]
        if text is None:
            continue

        ids = tokenizer.encode(text).ids

        all_ids.extend(ids)
        all_ids.append(end_token_id)

    arr = np.array(
        all_ids,
        dtype=dtype,
    )

    arr.tofile(output_path)

    print(
        f"Saved {output_path} "
        f"with {len(arr):,} tokens, "
        f"dtype={arr.dtype}"
    )


def prepare_tinystories(
    output_dir: str,
    vocab_size: int,
    dtype: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dtype == "uint16":
        np_dtype = np.uint16
    elif dtype == "uint32":
        np_dtype = np.uint32
    else:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Use uint16 or uint32."
        )

    if vocab_size > np.iinfo(np_dtype).max:
        raise ValueError(
            f"vocab_size={vocab_size} does not fit into {dtype}"
        )

    print("Loading TinyStories...")

    train_dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
    )

    val_dataset = load_dataset(
        "roneneldan/TinyStories",
        split="validation",
    )

    text_column = "text"

    print("Training byte-level BPE tokenizer...")

    tokenizer = train_bpe_tokenizer(
        train_dataset=train_dataset,
        vocab_size=vocab_size,
        text_column=text_column,
        output_dir=output_dir,
    )

    print("Encoding train split...")

    encode_dataset_to_bin(
        dataset=train_dataset,
        tokenizer=tokenizer,
        output_path=output_dir / "train.bin",
        text_column=text_column,
        dtype=np_dtype,
    )

    print("Encoding validation split...")

    encode_dataset_to_bin(
        dataset=val_dataset,
        tokenizer=tokenizer,
        output_path=output_dir / "val.bin",
        text_column=text_column,
        dtype=np_dtype,
    )

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tinystories",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32"],
    )

    return parser.parse_args()


def main():
    args = parse_args()

    prepare_tinystories(
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()