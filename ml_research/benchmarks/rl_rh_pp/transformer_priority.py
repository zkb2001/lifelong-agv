"""Transformer encoder + autoregressive pointer decoder (RL-RH-PP style)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .features import AGV_DIM, MAX_AGV, MAX_TASK, TASK_DIM


class PriorityTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        n_enc: int = 2,
        n_dec: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.agv_in = nn.Linear(AGV_DIM, d_model)
        self.task_in = nn.Linear(TASK_DIM, d_model)
        self.type_agv = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.type_task = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)
        self.query0 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.out = nn.Linear(d_model, d_model)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def encode(self, agv_f, agv_m, task_f, task_m):
        ae = self.agv_in(agv_f) + self.type_agv
        te = self.task_in(task_f) + self.type_task
        mem = torch.cat([ae, te], dim=1)
        pad = torch.cat([~agv_m, ~task_m], dim=1)
        mem = self.encoder(mem, src_key_padding_mask=pad)
        task_mem = mem[:, MAX_AGV : MAX_AGV + MAX_TASK, :]
        return mem, task_mem, pad

    def _pointer_logits(self, q, task_mem, avail):
        qh = self.out(q)
        scores = torch.einsum("bqd,btd->bqt", qh, task_mem).squeeze(1) / math.sqrt(
            self.d_model
        )
        return scores.masked_fill(~avail, -1e9)

    def forward_bc(self, agv_f, agv_m, task_f, task_m, teacher_idx) -> Dict[str, torch.Tensor]:
        mem, task_mem, pad = self.encode(agv_f, agv_m, task_f, task_m)
        B, L = teacher_idx.shape
        device = agv_f.device
        avail = task_m.clone()
        logits_list = []
        chosen = []
        q = self.query0.expand(B, 1, -1)
        for t in range(L):
            if chosen:
                tgt = torch.cat(chosen, dim=1)
                q = self.decoder(
                    tgt,
                    mem,
                    tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                        tgt.size(1), device=device
                    ),
                    memory_key_padding_mask=pad,
                )[:, -1:, :]
            logits = self._pointer_logits(q, task_mem, avail)
            logits_list.append(logits)
            idx = teacher_idx[:, t].clamp(min=0)
            gather = task_mem.gather(1, idx.view(B, 1, 1).expand(B, 1, self.d_model))
            chosen.append(gather)
            for b in range(B):
                j = int(teacher_idx[b, t].item())
                if j >= 0:
                    avail[b, j] = False
        logits_seq = torch.stack(logits_list, dim=1)
        pooled = mem.masked_fill(pad.unsqueeze(-1), 0.0).sum(1) / (
            (~pad).float().sum(1, keepdim=True).clamp(min=1.0)
        )
        value = self.value_head(pooled).squeeze(-1)
        return {"logits": logits_seq, "value": value}

    def decode_order(
        self,
        agv_f,
        agv_m,
        task_f,
        task_m,
        *,
        deterministic: bool = True,
        max_steps: Optional[int] = None,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        mem, task_mem, pad = self.encode(agv_f, agv_m, task_f, task_m)
        B = agv_f.size(0)
        device = agv_f.device
        avail = task_m.clone()
        n_valid = int(task_m.sum(dim=1).max().item())
        steps = max_steps or max(1, n_valid)
        orders: List[List[int]] = [[] for _ in range(B)]
        logps = []
        chosen = []
        q = self.query0.expand(B, 1, -1)
        for _ in range(steps):
            if not bool(avail.any()):
                break
            if chosen:
                tgt = torch.cat(chosen, dim=1)
                q = self.decoder(
                    tgt,
                    mem,
                    tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                        tgt.size(1), device=device
                    ),
                    memory_key_padding_mask=pad,
                )[:, -1:, :]
            logits = self._pointer_logits(q, task_mem, avail)
            if deterministic:
                idx = logits.argmax(dim=-1)
                logps.append(torch.zeros(B, device=device))
            else:
                dist = torch.distributions.Categorical(logits=logits)
                idx = dist.sample()
                logps.append(dist.log_prob(idx))
            gather = task_mem.gather(1, idx.view(B, 1, 1).expand(B, 1, self.d_model))
            chosen.append(gather)
            for b in range(B):
                j = int(idx[b].item())
                if avail[b, j]:
                    orders[b].append(j)
                    avail[b, j] = False
        lp = torch.stack(logps, dim=1).sum(dim=1) if logps else torch.zeros(B, device=device)
        return orders, lp

    def value_of(self, agv_f, agv_m, task_f, task_m) -> torch.Tensor:
        mem, _, pad = self.encode(agv_f, agv_m, task_f, task_m)
        pooled = mem.masked_fill(pad.unsqueeze(-1), 0.0).sum(1) / (
            (~pad).float().sum(1, keepdim=True).clamp(min=1.0)
        )
        return self.value_head(pooled).squeeze(-1)
