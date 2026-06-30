# generate.py

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformer.module import MyLLM


def top_p_filtering(
    probs: torch.Tensor,
    top_p: float | None,
) -> torch.Tensor:
    """
    Apply nucleus / top-p filtering.

    Args:
        probs: Tensor of shape (batch, vocab_size)
        top_p: Keep the smallest set of tokens whose cumulative probability >= top_p.

    Returns:
        Filtered and renormalized probabilities.
    """
    if top_p is None or top_p >= 1.0:
        return probs

    if top_p <= 0.0:
        raise ValueError(f"top_p must be > 0, got {top_p}")

    sorted_probs, sorted_indices = torch.sort(
        probs,
        dim=-1,
        descending=True,
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    remove_mask = cumulative_probs > top_p

    # Keep the first token that crosses top_p.
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(
        remove_mask,
        0.0,
    )

    filtered_probs = torch.zeros_like(probs)

    filtered_probs.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_probs,
    )

    filtered_probs = filtered_probs / filtered_probs.sum(
        dim=-1,
        keepdim=True,
    )

    return filtered_probs


@torch.no_grad()
def sample_next_token(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    Sample one next token.

    Args:
        model: Language model.
        input_ids: LongTensor of shape (batch, seq).
        temperature: Temperature for softmax. temperature=0 means greedy decoding.
        top_p: Nucleus sampling threshold.

    Returns:
        next_token: LongTensor of shape (batch,)
    """
    logits = model(input_ids)

    # Use the final position to predict the next token.
    next_logits = logits[:, -1, :]

    if temperature == 0:
        return torch.argmax(next_logits, dim=-1)

    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    next_logits = next_logits / temperature

    probs = torch.softmax(
        next_logits,
        dim=-1,
    )

    probs = top_p_filtering(
        probs,
        top_p,
    )

    next_token = torch.multinomial(
        probs,
        num_samples=1,
    )

    return next_token.squeeze(-1)


@torch.no_grad()
def generate_token_ids(
    model: torch.nn.Module,
    prompt_ids: list[int] | torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    end_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate token IDs from a prompt.

    Args:
        model: Language model.
        prompt_ids: list[int], Tensor shape (seq,), or Tensor shape (batch, seq).
        max_new_tokens: Maximum number of new tokens to generate.
        context_length: Model context length.
        end_token_id: Stop when this token is generated.
        temperature: Softmax temperature. 0 means greedy decoding.
        top_p: Nucleus sampling threshold.
        device: Device string.

    Returns:
        output_ids: LongTensor of shape (batch, total_seq_len)
    """
    model.eval()

    if isinstance(prompt_ids, list):
        input_ids = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=device,
        )
    else:
        input_ids = prompt_ids.to(
            device=device,
            dtype=torch.long,
        )

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    for _ in range(max_new_tokens):
        model_input = input_ids[:, -context_length:]

        next_token = sample_next_token(
            model=model,
            input_ids=model_input,
            temperature=temperature,
            top_p=top_p,
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token[:, None],
            ],
            dim=1,
        )

        if end_token_id is not None:
            if torch.all(next_token == end_token_id):
                break

    return input_ids


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    vocab_size: int,
    d_model: int,
    context_length: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
) -> MyLLM:
    model = MyLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    return model


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    end_token: str = "<|endoftext|>",
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str = "cpu",
) -> str:
    prompt_ids = tokenizer.encode(prompt)

    end_token_id = None
    if end_token is not None:
        end_ids = tokenizer.encode(end_token)
        if len(end_ids) != 1:
            raise ValueError(
                f"end_token should encode to one token, got {end_ids}"
            )
        end_token_id = end_ids[0]

    output_ids = generate_token_ids(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        end_token_id=end_token_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )

    return tokenizer.decode(
        output_ids[0].tolist()
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--end-token", type=str, default="<|endoftext|>")

    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    # 這裡假設你有自己的 tokenizer loading function。
    # 例如：
    #
    # from tokenizer import BPETokenizer
    # tokenizer = BPETokenizer.from_files(...)
    #
    # 請換成你自己的 tokenizer 初始化方式。
    raise NotImplementedError(
        "Please initialize your tokenizer here, then call generate_text(...)."
    )

    # text = generate_text(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prompt=args.prompt,
    #     max_new_tokens=args.max_new_tokens,
    #     context_length=args.context_length,
    #     end_token=args.end_token,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     device=args.device,
    # )
    #
    # print(text)


if __name__ == "__main__":
    main()