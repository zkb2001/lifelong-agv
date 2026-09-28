"""Emit canvas TSX with embedded compare_100 metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

CSV = Path(r"d:\compitition\MioVerse\final_version\ml_research\results\coord_custom_ai\compare_100\COMPARE_FULL.csv")
OUT = Path(r"C:\Users\zkb17\.cursor\projects\d-compitition-MioVerse-final-version\canvases\compare-100-metrics.canvas.tsx")

AGG = [
    {"method": "PIBT", "full": "8/20", "valid": "20/20", "avg_done": 63.9, "avg_sim": 62811, "avg_wall": 40.5, "sum_travel": 411561, "avg_travel": 20578},
    {"method": "M0", "full": "4/20", "valid": "20/20", "avg_done": 44.6, "avg_sim": 942, "avg_wall": 1478.5, "sum_travel": 40675, "avg_travel": 2034},
    {"method": "PP", "full": "1/20", "valid": "7/20", "avg_done": 32.4, "avg_sim": 848, "avg_wall": 1716.1, "sum_travel": 27138, "avg_travel": 1357},
    {"method": "ECBS", "full": "20/20", "valid": "18/20", "avg_done": 100.0, "avg_sim": 7280, "avg_wall": 549.7, "sum_travel": 158292, "avg_travel": 7915},
    {"method": "Hier", "full": "20/20", "valid": "20/20", "avg_done": 100.0, "avg_sim": 4584, "avg_wall": 686.0, "sum_travel": 118361, "avg_travel": 5918},
]

rows = []
for r in csv.DictReader(CSV.open(encoding="utf-8")):
    rows.append(
        {
            "slot": int(r["slot"]),
            "method": r["method"],
            "done": f"{r['done']}/{r['total']}",
            "completion": round(float(r["completion"]) * 100, 1),
            "valid": "Y" if r["valid"] == "True" else "N",
            "sim": int(r["sim_time"]),
            "wall": round(float(r["wall_s"]), 1),
            "travel": int(float(r["travel"])) if r["travel"] else None,
        }
    )

agg_json = json.dumps(AGG, ensure_ascii=False)
rows_json = json.dumps(rows, ensure_ascii=False)

tsx = """import {
  Divider,
  H1,
  H2,
  Stack,
  Table,
  Text,
} from "cursor/canvas";

const AGG = __AGG__ as const;
const ROWS = __ROWS__ as const;

export default function Compare100Metrics() {
  return (
    <Stack gap={16} style={{ padding: 16 }}>
      <H1>Compare 100 — 五方法全量对比</H1>
      <Text tone="secondary" size="small">
        SH01–20 × 100 tasks · travel = 全车队连续时刻曼哈顿移动格数之和 · Source: compare_100/results.jsonl + traj CSVs
      </Text>

      <H2>方法汇总</H2>
      <Table
        stickyHeader
        striped
        headers={[
          "Method",
          "full100",
          "VALID",
          "avg_done",
          "avg_sim",
          "avg_wall_s",
          "sum_travel",
          "avg_travel",
        ]}
        columnAlign={["left", "right", "right", "right", "right", "right", "right", "right"]}
        rows={AGG.map((a) => [
          a.method,
          a.full,
          a.valid,
          a.avg_done.toFixed(1),
          String(a.avg_sim),
          a.avg_wall.toFixed(1),
          a.sum_travel.toLocaleString(),
          a.avg_travel.toLocaleString(),
        ])}
      />

      <Divider />
      <H2>逐图 × 方法明细（100 行）</H2>
      <Text tone="secondary" size="small">
        列：完成度 / VALID / sim_time / 墙钟(s) / 总行驶距离
      </Text>
      <Table
        stickyHeader
        striped
        framed
        style={{ maxHeight: 640 }}
        headers={["Slot", "Method", "done", "valid", "sim_time", "wall_s", "travel"]}
        columnAlign={["left", "left", "right", "center", "right", "right", "right"]}
        rows={ROWS.map((r) => [
          `SH${String(r.slot).padStart(2, "0")}`,
          r.method,
          r.done,
          r.valid,
          String(r.sim),
          r.wall.toFixed(1),
          r.travel == null ? "NA" : r.travel.toLocaleString(),
        ])}
        rowTone={ROWS.map((r) =>
          r.valid === "N" ? "danger" : r.completion >= 100 ? "success" : undefined
        )}
      />
      <Text tone="secondary" size="small">
        文件：ml_research/results/coord_custom_ai/compare_100/COMPARE_FULL.md
      </Text>
    </Stack>
  );
}
"""

OUT.write_text(
    tsx.replace("__AGG__", agg_json).replace("__ROWS__", rows_json),
    encoding="utf-8",
)
print(f"wrote {OUT} rows={len(rows)}")
