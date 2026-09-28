"""Behavioral cloning for ParkArterySemaphoreNet (teacher labels)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import CKPT_PATH, GRID, N_MAP_CHANNELS, ensure_dirs
from .features import build_park_artery_channels
from .net import ParkArterySemaphoreNet
from .policy import save_ckpt
from .teacher import teacher_park_artery_from_channels

Cell = Tuple[int, int]


def _empty_walk_channels(
    *,
    failed_pos: Optional[Sequence[Cell]] = None,
    unloads: Optional[Sequence[Cell]] = None,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ch = np.zeros((N_MAP_CHANNELS, GRID, GRID), dtype=np.float32)
    ch[1, 1:21, 1:21] = 1.0
    # random sparse obstacles
    for _ in range(int(rng.integers(0, 8))):
        x, y = int(rng.integers(1, 21)), int(rng.integers(1, 21))
        ch[0, y, x] = 1.0
        ch[1, y, x] = 0.0
    for c in failed_pos or ():
        x, y = int(c[0]), int(c[1])
        if ch[1, y, x] > 0.5:
            ch[4, y, x] = 1.0
            ch[2, y, x] = 1.0
    # unload attractor
    for c in unloads or ():
        x, y = int(c[0]), int(c[1])
        for yy in range(GRID):
            for xx in range(GRID):
                if ch[1, yy, xx] < 0.5:
                    continue
                d = abs(xx - x) + abs(yy - y)
                ch[5, yy, xx] = max(ch[5, yy, xx], float(np.exp(-0.35 * d)))
    # random active path corridor
    if rng.random() < 0.5:
        y0 = int(rng.integers(1, 21))
        for x in range(1, 21):
            if ch[1, y0, x] > 0.5 and rng.random() < 0.7:
                ch[3, y0, x] = 1.0
    # congestion / topology filled lightly
    from .features import topology_degree_field

    ch[10] = topology_degree_field(ch[1])
    return ch


def make_synthetic_batch(
    batch_size: int = 8,
    *,
    seed: int = 0,
) -> Tuple[torch.Tensor, dict]:
    rng = np.random.default_rng(seed)
    xs = []
    park, art, conn, sem, feat = [], [], [], [], []
    for i in range(batch_size):
        unload = (int(rng.integers(2, 19)), int(rng.integers(2, 19)))
        fpos = (int(rng.integers(2, 19)), int(rng.integers(2, 19)))
        ch = _empty_walk_channels(
            failed_pos=[fpos], unloads=[unload], seed=int(seed) + i
        )
        lab = teacher_park_artery_from_channels(
            ch, goals=[unload], sem_pos=fpos, sem_goal=unload,
            pad_free=float(rng.random() > 0.3),
            path_clear=float(rng.random() > 0.3),
        )
        xs.append(ch)
        park.append(lab["parking"])
        art.append(lab["artery"])
        conn.append(lab["connectivity"])
        sem.append(lab["semaphore"])
        feat.append(lab["sem_feat"])
    x = torch.as_tensor(np.stack(xs), dtype=torch.float32)
    y = {
        "parking": torch.as_tensor(np.stack(park), dtype=torch.float32),
        "artery": torch.as_tensor(np.stack(art), dtype=torch.float32),
        "connectivity": torch.as_tensor(np.stack(conn), dtype=torch.float32),
        "semaphore": torch.as_tensor(np.asarray(sem), dtype=torch.float32),
        "sem_feat": torch.as_tensor(np.stack(feat), dtype=torch.float32),
    }
    return x, y


def load_replay_npz(path: Path) -> List[dict]:
    path = Path(path)
    if path.is_dir():
        from .freq_probe import load_freq_samples

        return load_freq_samples(path)
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        return list(data["samples"])
    return list(data)


def _batch_from_stored(samples: Sequence[dict], idx: np.ndarray, device) -> Tuple:
    """Use stored freq/teacher labels when present (no geometric overwrite)."""
    xs, park, art, conn, sem, feat = [], [], [], [], [], []
    for j in idx:
        s = samples[int(j)]
        ch = np.asarray(s["channels"], dtype=np.float32)
        if "parking" in s and "artery" in s and "connectivity" in s:
            xs.append(ch)
            park.append(np.asarray(s["parking"], dtype=np.float32))
            art.append(np.asarray(s["artery"], dtype=np.float32))
            conn.append(np.asarray(s["connectivity"], dtype=np.float32))
            sem.append(float(s.get("semaphore", 0.0)))
            feat.append(np.asarray(s["sem_feat"], dtype=np.float32))
            continue
        lab = teacher_park_artery_from_channels(
            ch,
            goals=s.get("goals") or [],
            sem_pos=tuple(s["sem_pos"]) if s.get("sem_pos") else None,
            sem_goal=tuple(s["sem_goal"]) if s.get("sem_goal") else None,
            pad_free=float(s.get("pad_free", 1.0)),
            path_clear=float(s.get("path_clear", 1.0)),
        )
        xs.append(ch)
        park.append(lab["parking"])
        art.append(lab["artery"])
        conn.append(lab["connectivity"])
        sem.append(lab["semaphore"])
        feat.append(lab["sem_feat"])
    x = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=device)
    return (
        x,
        torch.as_tensor(np.stack(park), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(art), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(conn), dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(sem), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(feat), dtype=torch.float32, device=device),
    )


def train_bc(
    *,
    steps: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
    replay_path: Optional[Path] = None,
    freq_path: Optional[Path] = None,
    out_ckpt: Optional[Path] = None,
    device: Optional[str] = None,
    freq_mix: float = 0.7,
) -> Path:
    ensure_dirs()
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = ParkArterySemaphoreNet().to(device_t)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    replay = load_replay_npz(Path(replay_path)) if replay_path else []
    freq_samples: List[dict] = []
    if freq_path:
        from .freq_probe import load_freq_samples

        freq_samples = load_freq_samples(Path(freq_path))
        print(f"[BC] freq samples={len(freq_samples)} from {freq_path}", flush=True)
    if freq_samples:
        # prefer frequency corpus; keep replay as optional mix
        pass

    net.train()
    for step in range(int(steps)):
        use_freq = bool(freq_samples) and (
            np.random.random() < float(freq_mix) or not replay
        )
        if use_freq:
            idx = np.random.randint(0, len(freq_samples), size=batch_size)
            x, y_park, y_art, y_conn, y_sem, sem_feat = _batch_from_stored(
                freq_samples, idx, device_t
            )
        elif replay and step % 3 == 0:
            idx = np.random.randint(0, len(replay), size=batch_size)
            x, y_park, y_art, y_conn, y_sem, sem_feat = _batch_from_stored(
                replay, idx, device_t
            )
        else:
            x, y = make_synthetic_batch(batch_size, seed=seed + step)
            x = x.to(device_t)
            y_park = y["parking"].to(device_t)
            y_art = y["artery"].to(device_t)
            y_conn = y["connectivity"].to(device_t)
            y_sem = y["semaphore"].to(device_t)
            sem_feat = y["sem_feat"].to(device_t)

        out = net(x, sem_feat=sem_feat)
        # parking is sparse → pos_weight
        pos = float(y_park.sum().clamp(min=1.0))
        neg = float((1.0 - y_park).sum().clamp(min=1.0))
        pw = torch.tensor([neg / pos], device=device_t)
        loss_p = F.binary_cross_entropy_with_logits(
            out["parking_logits"][:, 0], y_park, pos_weight=pw
        )
        loss_a = F.binary_cross_entropy_with_logits(out["artery_logits"][:, 0], y_art)
        loss_c = F.binary_cross_entropy_with_logits(
            out["connectivity_logits"][:, 0], y_conn
        )
        loss_s = F.binary_cross_entropy_with_logits(out["semaphore_logits"], y_sem)
        loss = loss_p + loss_a + loss_c + loss_s
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step + 1 == steps:
            print(
                f"[BC] step={step+1}/{steps} loss={float(loss):.4f} "
                f"p={float(loss_p):.3f} a={float(loss_a):.3f} "
                f"c={float(loss_c):.3f} s={float(loss_s):.3f}",
                flush=True,
            )

    path = Path(out_ckpt or CKPT_PATH)
    return save_ckpt(net, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", type=str, default="")
    ap.add_argument(
        "--freq",
        type=str,
        default="",
        help="freq_probe run dir or all_samples.npz (preferred BC labels)",
    )
    ap.add_argument("--freq-mix", type=float, default=0.7)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    train_bc(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        replay_path=Path(args.replay) if args.replay else None,
        freq_path=Path(args.freq) if args.freq else None,
        freq_mix=float(args.freq_mix),
        out_ckpt=Path(args.out) if args.out else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
