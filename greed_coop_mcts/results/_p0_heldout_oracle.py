"""Build SH01–20-only oracle JSON and train facn_p0_heldout.pt."""
from __future__ import annotations

import json
from pathlib import Path

from greed_coop_mcts.paths import CKPT, RESULTS

SRC = RESULTS / "facn_oracle_sweep_v2.json"
OUT_ORACLE = RESULTS / "facn_oracle_heldout_01_20.json"


def main() -> int:
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    best = {k: v for k, v in (raw.get("best") or {}).items() if 1 <= int(k) <= 20}
    delta = {k: v for k, v in (raw.get("delta_star") or {}).items() if 1 <= int(k) <= 20}
    geoms = {k: v for k, v in (raw.get("geoms") or {}).items() if 1 <= int(k) <= 20}
    out = {
        "best": best,
        "delta_star": delta,
        "geoms": geoms,
        "note": "P0 held-out train: SH01-20 only; SH21-40 are test",
    }
    OUT_ORACLE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT_ORACLE} n={len(best)}")
    print(f"ckpt_target {CKPT / 'facn_p0_heldout.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
