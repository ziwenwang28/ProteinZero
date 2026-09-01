# structure.py
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pickle
from typing import List, Sequence, Optional, Tuple

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

import traceback

# Gated linear unit modules.
class GLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.act1 = nn.GELU()
        self.act2 = F.silu
        self.dense_h_to_4h = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.dense_4h_to_h = nn.Linear(hidden_size * 4, hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x

def swiglu(x):
    """PaLM-style SwiGLU activation"""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return F.silu(x1) * x2

class GLU_new(nn.Module):
    """A simplified GLU block with optional dropout."""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        intermediate_size = 1280  # Fixed intermediate width.
        self.act = swiglu
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.act(x)
        x = self.dense_4h_to_h(x)
        x = self.dropout(x)
        return x

# Sinusoidal positional embeddings for the resampler.
def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

# Multi-head attention resampler for structure embeddings.
class Resampler(nn.Module):
    def __init__(
        self,
        kv_dim,
        embed_dim,
        num_heads=8,
        n_queries=64,
        max_seqlen=1024,
        perceiver_resampler_positional_emb=True,
        use_GLU=False,
        bos_init=False,
        dropout=0.0
    ):
        super().__init__()
        self.perceiver_resampler_positional_emb = perceiver_resampler_positional_emb
        self.n_queries = n_queries
        self.max_seqlen = max_seqlen

        if self.perceiver_resampler_positional_emb:
            # Sample positions in [0, max_seqlen).
            self.stride = max_seqlen // n_queries
            pos = np.arange(max_seqlen, dtype=np.float32)
            pos_emb = get_1d_sincos_pos_embed(embed_dim, pos)
            self.register_buffer("pos_embed", torch.from_numpy(pos_emb).float())

        # Learnable latent queries.
        self.latents = nn.Parameter(torch.randn(n_queries, embed_dim))
        if bos_init:
            # Retain the initial standard-normal latent values.
            pass
        else:
            nn.init.trunc_normal_(self.latents, std=1e-3)

        # Project input features from kv_dim to embed_dim.
        self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)

        # Multi-head attention.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)

        if use_GLU:
            print("Resampler using GLU_new.")
            self.proj = GLU_new(embed_dim, dropout=dropout)
        else:
            self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, struc_x):
        """
        struc_x: dict with:
          'encoder_out' shape (B, L, kv_dim)
          'encoder_padding_mask' shape (B, L) -> True=1 => "valid"
        Returns:
          shape (B, n_queries, embed_dim)
        """
        x = struc_x["encoder_out"]                # (B, L, kv_dim)
        mask = struc_x["encoder_padding_mask"]    # (B, L); True denotes a valid position.

        # Replace NaN values with zeros.
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            x = x.masked_fill(nan_mask, 0.0)

        # Project input features to embed_dim.
        x = self.kv_proj(x)
        x = self.ln_kv(x)    # (B, L, embed_dim)

        b, seqlen = x.shape[:2]
        latents = self.ln_q(self.latents)  # (n_queries, embed_dim)

        # Add positional embeddings to latent queries and input tokens.
        if self.perceiver_resampler_positional_emb:
            latents = latents + self.pos_embed[::self.stride][: self.n_queries]
            x = x + self.pos_embed[:seqlen]

        # Replicate latent queries across the batch.
        latents = repeat(latents, "n d -> b n d", b=b)

        # Match the attention projection dtype.
        common_dtype = self.attn.in_proj_weight.dtype
        latents = latents.to(common_dtype)
        x = x.to(common_dtype)

        # Multi-head attention uses True for padded keys, whereas the input mask
        # uses True for valid positions.
        key_padding_mask = ~mask.bool()

        out = self.attn(
            query=latents,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask  # (B, L)
        )[0]

        out = self.ln_post(out)
        out = self.proj(out)   # (B, n_queries, embed_dim)
        return out

# Structure embedding encoder.
class StructureTransformer(nn.Module):
    """
    Given a structure ID (like "ref1"), we read from
    structure_embeddings/ref1, parse 'mpnn_emb' or 'pifold_emb', etc.,
    pad them to max_seqlen, then run them through the Resampler.
    """
    def __init__(
        self,
        width: int = 640,
        n_queries: int = 32,
        output_dim: int = 4096,
        embedding_keys=set(["mpnn_emb","pifold_emb"]),
        max_seqlen: int = 1024,
        num_heads: int = 8,
        structure_emb_path_prefix='structure_embeddings',
        **kwargs
    ):
        super().__init__()
        self.structure_emb_path_prefix = structure_emb_path_prefix
        self.embedding_keys = embedding_keys
        self.max_seqlen = max_seqlen
        self.width = width
        self.n_queries = n_queries

        self.attn_pool = Resampler(
            embed_dim=output_dim,
            kv_dim=width,
            n_queries=n_queries,
            max_seqlen=max_seqlen,
            num_heads=num_heads,
            **kwargs
        )

    def prepare_structure(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample: a dict loaded from e.g. 'ref1.pyd'
        Return:
          emb_pad (max_seqlen, self.width)
          emb_mask (max_seqlen,) with True where valid
        """
        # Allocate fixed-size output buffers.
        emb_pad  = torch.zeros((self.max_seqlen, self.width))
        emb_mask = torch.zeros((self.max_seqlen,), dtype=bool)

        # Restore masked PiFold positions as NaNs.
        if "pifold_emb" in self.embedding_keys and "pifold_mask" in sample:
            mask = sample["pifold_mask"]            # (L,)
            pifo = sample["pifold_emb"]             # (N, feature_dim)
            filled = pifo.new_full((mask.shape[0], pifo.shape[1]), float("nan"))
            filled[mask > 0] = pifo
            sample["pifold_emb"] = filled

        # Collect requested embeddings and convert them to tensors.
        emb_list = []
        for ek in self.embedding_keys:
            if ek not in sample:
                continue
            data = sample[ek]

            # Concatenate list-valued embeddings along the sequence dimension.
            if isinstance(data, list):
                if data and isinstance(data[0], np.ndarray):
                    data = np.concatenate(data, axis=0)
                else:
                    data = torch.cat(data, dim=0)

            # Convert NumPy arrays to tensors.
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            elif not torch.is_tensor(data):
                raise TypeError(f"{ek} has unsupported type {type(data)}")

            if not torch.is_floating_point(data):
                data = data.float()

            emb_list.append(data)

        # Concatenate embeddings along the feature dimension.
        emb = torch.cat(emb_list, dim=-1) if emb_list else torch.zeros((0, ), dtype=torch.float32)

        # Match the configured feature width by truncation or zero-padding.
        if emb.shape[1] > self.width:                       # Truncate excess features.
            emb = emb[:, : self.width]
        elif emb.shape[1] < self.width:                     # Zero-pad missing features.
            pad_cols = self.width - emb.shape[1]
            emb = torch.cat([emb, emb.new_zeros(emb.shape[0], pad_cols)], dim=1)

        # Copy the sequence into fixed-length output buffers.
        length = min(emb.shape[0], self.max_seqlen)
        emb_pad[:length]  = emb[:length]
        emb_mask[:length] = True

        return emb_pad, emb_mask

    def forward(self, x: dict):
        """
        x is a dict:
          {
             'encoder_out': (B, L, width),
             'encoder_padding_mask': (B, L)
          }
        We'll run self.attn_pool, returning (B, n_queries, output_dim).
        """
        return self.attn_pool(x)

    def encode(self, structure_paths: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        structure_paths: a list of 1D Tensors with ASCII codes, for each structure ID.
        E.g. "ref1|..." => [114,101,102,49,124,...].
        We'll parse that, find the part before '|', load {structure_id}.pyd, then run them.
        Return shape => (B, n_queries, output_dim) or None if a file is missing.
        """

        if not hasattr(self, "_printed_ids"):
            self._printed_ids = set()

        structure_embs = []
        structure_masks = []

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        for path_tokens in structure_paths:

            # Decode the ASCII structure identifier.
            path_str = "".join(chr(c) for c in path_tokens[: self.n_queries].tolist() if c > 0)

            # Retain the component preceding the delimiter.
            pipe_idx = path_str.find("|")
            if pipe_idx != -1:
                path_str = path_str[:pipe_idx]

            path_str = path_str.strip()

            DEBUG_PRINT = False

            if local_rank == 0 and path_str not in self._printed_ids and DEBUG_PRINT:

                self._printed_ids.add(path_str)

            full_path = os.path.join(self.structure_emb_path_prefix, path_str)
            if not os.path.exists(full_path):

                return None

            with open(full_path, "rb") as f:
                sample = pickle.load(f)
                structure, struc_mask = self.prepare_structure(sample)

            structure_embs.append(structure)
            structure_masks.append(struc_mask)

        device_ = next(self.parameters()).device
        dtype_ = next(self.parameters()).dtype

        structure_embs = torch.stack(structure_embs, dim=0).to(device=device_, dtype=dtype_)
        structure_masks = torch.stack(structure_masks, dim=0).to(device=device_)

        # Pool the prepared structure embeddings.
        out = self({
            "encoder_out": structure_embs,          # (B, L, width)
            "encoder_padding_mask": structure_masks # (B, L)
        })
        return out
