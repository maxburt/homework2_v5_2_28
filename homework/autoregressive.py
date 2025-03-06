import abc

import torch
import torch.nn as nn
import torch.nn.functional as F



def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(Autoregressive, torch.nn.Module):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_heads: int = 8, n_layers: int = 6):
        super().__init__()
        #raise NotImplementedError()

        self.n_tokens = n_tokens
        self.d_latent = d_latent

        # Embedding for input tokens
        self.embedding = nn.Embedding(n_tokens, d_latent)

        # Transformer Encoder Layer (Causal structure)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection layer
        self.output_layer = nn.Linear(d_latent, n_tokens)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        #raise NotImplementedError()

        B, H, W = x.shape  # (Batch, Height, Width)

        # Flatten spatial dimensions into a sequence
        x = x.view(B, H * W)  # Shape: (B, H*W)

        # Shift input sequence right (prepend start token, remove last token)
        shifted_x = torch.cat(
            [torch.full((B, 1), self.n_tokens - 1, dtype=torch.long, device=x.device), x[:, :-1]], dim=1
        )

        # Embed tokens
        x = self.embedding(shifted_x)  # Shape: (B, H*W, d_latent)

        # Create causal mask (Ensure no token sees future tokens)
        seq_len = x.shape[1]
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)

        # Pass through transformer
        x = self.transformer(x, mask=mask)

        # Output logits
        x = self.output_layer(x)  # Shape: (B, H*W, n_tokens)

        # Reshape back to image format
        x = x.view(B, H, W, self.n_tokens)  # Shape: (B, H, W, n_tokens)

        return x, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        #raise NotImplementedError()
        device = device or next(self.parameters()).device
        output = torch.full((B, h, w), self.n_tokens - 1, dtype=torch.long, device=device)  # Initialize with start token

        for i in range(h):
            for j in range(w):
                # Flatten and shift input
                flattened_output = output.view(B, -1)
                shifted_input = torch.cat(
                    [torch.full((B, 1), self.n_tokens - 1, dtype=torch.long, device=device), flattened_output[:, :-1]],
                    dim=1,
                )

                # Embed tokens
                embedded = self.embedding(shifted_input)

                # Apply causal mask
                seq_len = embedded.shape[1]
                mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

                # Pass through transformer
                transformer_output = self.transformer(embedded, mask=mask)

                # Predict token at (i, j)
                logits = self.output_layer(transformer_output)
                probs = F.softmax(logits[:, -1, :], dim=-1)  # Get last token prediction

                # Sample from probability distribution
                sampled_token = torch.multinomial(probs, 1).squeeze(-1)

                # Set predicted token in output image
                output[:, i, j] = sampled_token

        return output