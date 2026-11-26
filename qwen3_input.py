# /// script
# requires-python = ">=3.10"
# dependencies = ["safetensors", "torch", "transformers==4.57.6"]
# ///
"""Turn plain text into Qwen3-8B's input embeddings."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

EMBEDDING_WEIGHT = "model.embed_tokens.weight"


class Qwen3Input:
    def __init__(self, checkpoint_dir: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, local_files_only=True
        )
        index = json.loads(
            (checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        shard_path = checkpoint_dir / index["weight_map"][EMBEDDING_WEIGHT]
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            weight = shard.get_tensor(EMBEDDING_WEIGHT)
        self.embedding = torch.nn.Embedding.from_pretrained(weight)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def embed_tokens(self, token_ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.embedding.weight.device
        )
        return self.embedding(input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path, help="Local Qwen3-8B checkpoint directory")
    parser.add_argument("text", nargs="?", default="Hello!")
    args = parser.parse_args()

    model_input = Qwen3Input(args.checkpoint_dir)
    token_ids = model_input.encode_text(args.text)
    embeddings = model_input.embed_tokens(token_ids)
    if token_ids:
        first_row = model_input.embedding.weight[token_ids[0]]
        torch.testing.assert_close(embeddings[0, 0], first_row, rtol=0, atol=0)


if __name__ == "__main__":
    main()
