import torch
from torch import nn
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)

from config import QVK_PROJ_BLOCKSIZE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


### Transformer Encoder Model
class BaseTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        dim_ff: int,
        n_classes: int,
        num_layers: int,
        n_heads: int,
        n_tokens: int,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=n_tokens)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model * n_tokens, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_embed = self.input_projection(x)
        x_embed_pe = self.positional_encoding(x_embed)

        z = self.encoder(x_embed_pe)

        # Flatten the output of the encoder
        z_flat = z.flatten(start_dim=1)

        logits = self.output_projection(z_flat)

        return logits


### Optuna-tunable xLSTM

class TunableXLSTMClassifier(nn.Module):
    """One xLSTM classifier covering all three block layouts.

    Optuna needs to move between layouts and head types inside one search
    space, so this class takes both as arguments:

      block_mix  "mlstm" -> every block is an mLSTM block
                 "mixed" -> mLSTM blocks with one sLSTM block last
                 "slstm" -> every block is an sLSTM block
      pooling    "last" / "mean" / "flatten" - how (B, T, d_model) collapses to
                 a single vector before the classification head.

    No positional encoding: xLSTM is recurrent and causal, so position is
    already implicit in the scan.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_classes: int,
        n_tokens: int,
        num_blocks: int,
        num_heads: int,
        conv1d_kernel_size: int,
        qkv_proj_blocksize: int,
        proj_factor: float,
        dropout: float,
        block_mix: str,
        pooling: str,
    ):
        super().__init__()
        self.pooling = pooling

        # "mixed" needs somewhere to put the mLSTM block; with a single block it
        # would silently collapse to an all-sLSTM stack, so name it as such.
        if block_mix == "mixed" and num_blocks < 2:
            block_mix = "mlstm"
        self.block_mix = block_mix

        self.input_projection = nn.Linear(input_size, d_model)

        slstm_at = {"mlstm": [], "mixed": [num_blocks - 1], "slstm": "all"}[block_mix]

        # Built fresh every time: xLSTMBlockStackConfig.__post_init__ mutates the
        # nested block configs (embedding_dim, dropout, context_length), so a
        # config object must never be shared across models or trials.
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=None if block_mix == "slstm" else mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size,
                    qkv_proj_blocksize=qkv_proj_blocksize,
                    num_heads=num_heads,
                    proj_factor=proj_factor,
                )
            ),
            slstm_block=None if block_mix == "mlstm" else sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    # Both are required on a CPU-only machine: sLSTMCellConfig
                    # defaults to backend="cuda" (which JIT-compiles CUDA
                    # kernels) and dtype="bfloat16".
                    backend="vanilla",
                    dtype="float32",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=n_tokens,  # sizes the mLSTM causal mask - must be the seq len
            num_blocks=num_blocks,
            embedding_dim=d_model,
            dropout=dropout,
            # Must be passed here, not assigned afterwards: __post_init__ turns
            # it into the block map that decides which block class goes where.
            slstm_at=slstm_at,
        )

        self.encoder = xLSTMBlockStack(xlstm_config)

        head_in = d_model * n_tokens if pooling == "flatten" else d_model
        self.output_projection = nn.Linear(head_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, input_size) -> (B, T, d_model)
        z = self.encoder(self.input_projection(x))

        if self.pooling == "last":
            z = z[:, -1, :]
        elif self.pooling == "mean":
            z = z.mean(dim=1)
        else:
            z = z.flatten(start_dim=1)

        return self.output_projection(z)


def buildModel(params, input_size, n_classes, n_tokens, device):
    """Single choke point for model construction.

    Called three times per run with the same params (trial, retrain, reload), so
    every architectural knob comes from params rather than module constants -
    otherwise the reloaded model would not match the saved state_dict.

    Dispatches on params["model_type"], which suggestHyperparams tags onto the
    dict. Defaults to "xlstm" when the tag is absent.
    """
    if params.get("model_type", "xlstm") == "transformer":
        return BaseTransformerClassifier(
            input_size=input_size,
            d_model=params["d_model"],
            dim_ff=params["dim_ff"],
            n_classes=n_classes,
            num_layers=params["num_layers"],
            n_heads=params["n_heads"],
            n_tokens=n_tokens,
        ).to(device)

    return TunableXLSTMClassifier(
        input_size=input_size,
        d_model=params["d_model"],
        n_classes=n_classes,
        n_tokens=n_tokens,
        num_blocks=params["num_blocks"],
        num_heads=params["num_heads"],
        conv1d_kernel_size=params["conv1d_kernel_size"],
        qkv_proj_blocksize=params.get("qkv_proj_blocksize", QVK_PROJ_BLOCKSIZE),
        proj_factor=params["proj_factor"],
        dropout=params["dropout"],
        block_mix=params["block_mix"],
        pooling=params["pooling"],
    ).to(device)
