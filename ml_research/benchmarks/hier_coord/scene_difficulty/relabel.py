"""Relabel samples with relaxed 'qualified' rule (completion-based).

Old: must finish within tight wall → many near-complete runs marked hard.
New: completion_ratio >= qualify_cr ⇒ easy (would finish in reasonable time);
     else hard (stuck / mostly incomplete).

Does NOT re-run M0; rewrites meta.json label fields in-place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ml_research.common.paths import RESULTS

DEFAULT_TRAIN = RESULTS / "hier_coord" / "scene_difficulty"
DEFAULT_TEST = RESULTS / "hier_coord" / "scene_difficulty_test"


def qualify_label(
    *,
    completion_ratio: float,
    forced_stop: bool,
    qualify_cr: float = 0.90,
) -> tuple[str, int, str]:
    """Return (label, label_int, rule)."""
    cr = float(completion_ratio)
    if cr >= 0.999 and not forced_stop:
        return "easy", 0, "finished"
    if cr >= float(qualify_cr):
        return "easy", 0, f"cr>={qualify_cr:g}_qualified"
    return "hard", 1, f"cr<{qualify_cr:g}_stuck"


def relabel_dir(samples_dir: Path, *, qualify_cr: float) -> dict:
    n = n_flip = n_easy = n_hard = 0
    for d in samples_dir.iterdir():
        meta_p = d / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        old = int(meta.get("label_int", -1))
        # keep raw M0 fields
        meta["label_strict"] = meta.get("label")
        meta["label_int_strict"] = meta.get("label_int")
        lab, li, rule = qualify_label(
            completion_ratio=float(meta.get("completion_ratio") or 0.0),
            forced_stop=bool(meta.get("forced_stop")),
            qualify_cr=qualify_cr,
        )
        meta["label"] = lab
        meta["label_int"] = li
        meta["label_rule"] = rule
        meta["qualify_cr"] = float(qualify_cr)
        if old != li:
            n_flip += 1
        if li == 0:
            n_easy += 1
        else:
            n_hard += 1
        n += 1
        meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"n": n, "n_flip": n_flip, "n_easy": n_easy, "n_hard": n_hard}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=str, default=str(DEFAULT_TRAIN))
    ap.add_argument("--test-data", type=str, default=str(DEFAULT_TEST))
    ap.add_argument(
        "--qualify-cr",
        type=float,
        default=0.90,
        help="completion_ratio >= this ⇒ easy (reasonable-time qualified)",
    )
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args(argv)

    for name, root in (("train", Path(args.train_data)), ("test", Path(args.test_data))):
        if name == "test" and args.skip_test:
            continue
        samples = root / "samples"
        if not samples.exists():
            print(f"[relabel] skip missing {samples}", flush=True)
            continue
        stats = relabel_dir(samples, qualify_cr=float(args.qualify_cr))
        print(
            f"[relabel] {name}: n={stats['n']} flip={stats['n_flip']} "
            f"easy={stats['n_easy']} hard={stats['n_hard']} "
            f"qualify_cr={args.qualify_cr}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
