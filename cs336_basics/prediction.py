from .basics import softmax, TransformerLanguageModel
from .bpe import Tokenizer
from .common_types import MergeList, Vocab

import torch
from torch import Tensor


class Predictor:
    model: TransformerLanguageModel
    tokenizer: Tokenizer
    nucleus: float
    temperature: float

    def __init__(
        self,
        model: TransformerLanguageModel,
        tokenizer: Tokenizer,
        temperature: float,
        nucleus: float,
        max_tokens: int,
        device: torch.device | None = None,
    ):
        model.eval()
        self.max_tokens = max_tokens
        self.model = model
        self.nucleus = nucleus
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.device = device

    def _computeNextTokenDistribution(self, input: torch.Tensor):
        # Get the logits for the last token in the sequence
        logits = self.model(input)[:, -1, :]
        # Apply temperature scaling
        logits.div_(self.temperature)
        # Compute probabilities using softmax
        probs = softmax(logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create a mask for tokens to remove.
        # We shift the cumulative probabilities to the right and check if they are greater than the nucleus value.
        # This ensures that we keep the first token that pushes the cumulative probability over the nucleus threshold.
        sorted_indices_to_remove = cumulative_probs > self.nucleus
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create a mask for the original indices
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            -1, sorted_indices, sorted_indices_to_remove
        )

        # Set probabilities of tokens to remove to 0
        probs[indices_to_remove] = 0

        # Re-normalize the probabilities
        probs.div_(probs.sum(dim=-1, keepdim=True))

        return probs

    def _sampleNextTokenFromDistribution(self, distribution: torch.Tensor) -> int | None:
        # Sample the probability distribution to get the next token
        if distribution.sum() == 0:
            return None
        next_token = torch.multinomial(distribution, num_samples=1)
        return next_token.item()

    def _predictTokens(self, prompt_tokens: torch.Tensor) -> list[int]:
        # Make a copy of the input to avoid modifying the original
        output_tokens = []
        while len(output_tokens) < self.max_tokens:
            # We can only pass in context_length tokens to the model
            context = prompt_tokens[:, -self.model.context_length :]
            # Pass the current full sequence to the model
            distribution = self._computeNextTokenDistribution(context)
            next_token_id = self._sampleNextTokenFromDistribution(distribution)

            if next_token_id is None:
                # Stop if sampling returns no token
                break

            # Append the predicted token ID to our list of outputs
            output_tokens.append(next_token_id)

            # Create a new tensor for the next token
            next_token_tensor = torch.tensor(
                [[next_token_id]], device=self.device, dtype=torch.long
            )

            # Concatenate the new token to the current sequence to form the input for the next iteration
            prompt_tokens = torch.cat([prompt_tokens, next_token_tensor], dim=1)

        return output_tokens

    def predict(self, input: str) -> str:
        sequence = torch.as_tensor(
            self.tokenizer.encode(input), device=self.device, dtype=torch.long
        ).unsqueeze(0)
        output_tokens = self._predictTokens(sequence)
        return self.tokenizer.decode(output_tokens)
