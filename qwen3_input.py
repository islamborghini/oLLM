# /// script
# requires-python = ">=3.10"
# dependencies = ["safetensors", "torch", "transformers==4.57.6"]
# ///
"""Prepare Qwen3-8B's input for attention in decoder block 0."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

EMBEDDING_WEIGHT = "model.embed_tokens.weight"
BLOCK0_NORM_WEIGHT = "model.layers.0.input_layernorm.weight"


def load_weight(checkpoint_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(checkpoint_dir / weight_map[name], framework="pt", device="cpu") as shard:
        return shard.get_tensor(name)


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    values = hidden_states.float()
    mean_square = values.square().mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(mean_square + eps)
    return weight * normalized.to(hidden_states.dtype)


class Qwen3Input(torch.nn.Module):
    def __init__(self, checkpoint_dir: Path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, local_files_only=True
        )
        config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        self.rms_norm_eps = config["rms_norm_eps"]
        weight_map = json.loads(
            (checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )["weight_map"]
        embedding_weight = load_weight(checkpoint_dir, weight_map, EMBEDDING_WEIGHT)
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_weight)
        norm_weight = load_weight(checkpoint_dir, weight_map, BLOCK0_NORM_WEIGHT)
        self.block0_norm_weight = torch.nn.Parameter(norm_weight, requires_grad=False)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def embed_tokens(self, token_ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.embedding.weight.device
        )
        return self.embedding(input_ids)

    def forward(self, token_ids: list[int]) -> torch.Tensor:
        embeddings = self.embed_tokens(token_ids)
        return rms_norm(embeddings, self.block0_norm_weight, self.rms_norm_eps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path, help="Local Qwen3-8B checkpoint directory")
    parser.add_argument("text", nargs="?", default="Hello!")
    args = parser.parse_args()

    model_input = Qwen3Input(args.checkpoint_dir)
    token_ids = model_input.encode_text(args.text)
    model_input(token_ids)


if __name__ == "__main__":
    main()
