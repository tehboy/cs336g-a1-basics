import argparse
import sys
import torch

from cs336_basics.bpe import Tokenizer
from cs336_basics.data import load_checkpoint
from cs336_basics.basics import TransformerLanguageModel
from cs336_basics.prediction import Predictor
from cs336_basics.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Generate text from a prompt.")
    parser.add_argument("--checkpoint-file", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--vocab-file", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--merges-file", type=str, required=True, help="Path to the merges file.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--nucleus", type=float, default=0.9, help="Nucleus for sampling.")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate.")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--d-ff", type=int, default=1344, help="Feedforward dimension.")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Torch dtype (e.g., float32, float16, bfloat16).",
    )
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint_file, map_location=device)

    model_state = checkpoint["model"]
    vocab_size = checkpoint["config"]["vocab_size"]

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=checkpoint["config"]["context_length"],
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        dtype=getattr(torch, args.dtype),
        device=device,
    )
    model.load_state_dict(model_state)

    tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file)

    predictor = Predictor(
        model=model,
        tokenizer=tokenizer,
        temperature=args.temperature,
        nucleus=args.nucleus,
        max_tokens=args.max_tokens,
        device=device,
    )

    print("Enter a prompt:")
    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:
            break
        output = predictor.predict(prompt)
        print(output)
        print("\nEnter a prompt:")

if __name__ == "__main__":
    main()
