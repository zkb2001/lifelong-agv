"""Re-validate SH21–40 gen trajectories and refresh GEN_CURRICULUM_EVAL.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from greed_coop_mcts.scenario import load_slot_meta  # noqa: E402
from greed_coop_mcts.validate import format_validation_summary, validate_trajectory  # noqa: E402

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "GEN_CURRICULUM_EVAL.json").read_text(encoding="utf-8"))

for r in rows:
    slot = int(r["slot"])
    p = ROOT / "trajectories" / f"SH{slot:02d}_gen_gcm.csv"
    if not p.exists():
        r["mot"] = -1
        r["vnote"] = "no_traj"
        continue
    meta = dict(load_slot_meta(slot) or {})
    meta["id"] = f"SH{slot:02d}_gen"
    meta["n_tasks"] = 100
    rep = validate_trajectory(meta, p)
    iss = rep.get("issues") or {}
    r["mot"] = int(iss.get("n_illegal_motion") or 0)
    r["coll"] = int(iss.get("n_collisions") or 0)
    r["swap"] = int(iss.get("n_swaps") or 0)
    r["fifo"] = int(iss.get("n_fifo_violations") or 0)
    r["validate_ok"] = bool(rep.get("ok"))
    r["ok"] = bool(r["validate_ok"] and r["done"] == 100)
    r["vnote"] = format_validation_summary(rep)
    print(f"SH{slot:02d} done={r['done']} {r['vnote']}", flush=True)

(ROOT / "GEN_CURRICULUM_EVAL.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
)

ok_n = sum(1 for r in rows if r["ok"])
done100 = sum(1 for r in rows if r["done"] == 100)
sims = [r["sim"] for r in rows if r["ok"]]
sims_done = [r["sim"] for r in rows if r["done"] == 100]
easy = [r for r in rows if r["difficulty"] == "easy"]
med = [r for r in rows if r["difficulty"] == "medium"]
hard = [r for r in rows if r["difficulty"] == "hard"]

lines = [
    "# Generated maps SH21–40 vs current rrsc stack",
    "",
    f"- maps: {len(rows)} (easy={len(easy)}, medium={len(med)}, hard={len(hard)})",
    "- all free-space 4-connected: **yes** (asserted at gen)",
    f"- complete 100/100: **{done100}/{len(rows)}**",
    f"- complete 100/100 VALID: **{ok_n}/{len(rows)}**",
    (
        f"- avg sim (VALID only): **{sum(sims) / len(sims):.1f}**"
        if sims
        else "- avg sim (VALID): n/a"
    ),
    (
        f"- avg sim (done=100): **{sum(sims_done) / len(sims_done):.1f}**"
        if sims_done
        else "- avg sim (done=100): n/a"
    ),
    "",
    "| Slot | diff | style | n | near | obs | sim | done | ok | mot | asa | cebi | e4 |",
    "|------|------|-------|---|------|-----|-----|------|----|-----|----|------|----|",
]
for r in rows:
    lines.append(
        f"| SH{r['slot']:02d} | {r['difficulty']} | {r['style']} | {r['n']} | "
        f"{r['near']} | {r['n_obs']} | {r['sim']} | {r['done']}/100 | {int(r['ok'])} | "
        f"{r['mot']} | {r['asa']} | {r['cebi']} | {r['e4']} |"
    )
fail = [r for r in rows if not r["ok"]]
if fail:
    lines += ["", "## Failures", ""]
    for r in fail:
        reason = (
            f"incomplete done={r['done']}"
            if r["done"] < 100
            else r.get("vnote", "INVALID")
        )
        lines.append(
            f"- SH{r['slot']:02d} [{r['difficulty']}/{r['style']}] n={r['n']} "
            f"near={r['near']}: {reason}"
        )

(ROOT / "GEN_CURRICULUM_EVAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"ok {ok_n}/{len(rows)}", flush=True)
