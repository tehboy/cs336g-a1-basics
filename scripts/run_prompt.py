import sys
import torch
from cs336_basics.utils import ModelArgs, get_device
from cs336_basics.basics import TransformerLanguageModel
from cs336_basics.bpe import Tokenizer
from cs336_basics.prediction import Predictor
from cs336_basics.data import load_checkpoint


def main():
    args = ModelArgs()

    # Determine the prompt
    prompt = ""
    if args.prompt:
        prompt = str(args.prompt)
    elif args.prompt_file:
        with open(str(args.prompt_file), "r") as f:
            prompt = f.read()
    else:
        prompt = sys.stdin.read()

    # Set up device and dtype
    device = get_device()
    dtype = getattr(torch, str(args.dtype))

    # Initialize model
    model = TransformerLanguageModel(
        vocab_size=int(args.vocab_size),
        context_length=int(args.context_length),
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        d_ff=int(args.d_ff),
        rope_theta=float(args.rope_theta),
        device=device,
        dtype=dtype,
    )
    model.to(device)

    # Load checkpoint
    if args.checkpoint_file:
        load_checkpoint(str(args.checkpoint_file), model)
    else:
        print("Warning: No checkpoint file provided. Using an untrained model.")

    # Initialize tokenizer
    tokenizer = Tokenizer.from_files(str(args.vocab_path), str(args.merges_path))

    # Initialize predictor
    predictor = Predictor(
        model=model,
        tokenizer=tokenizer,
        temperature=float(args.temperature),
        nucleus=float(args.nucleus),
        max_tokens=int(args.max_tokens),
        device=device,
    )

    # Generate and print the prediction
    generated_text = predictor.predict(prompt)
    print(generated_text)


if __name__ == "__main__":
    main()
