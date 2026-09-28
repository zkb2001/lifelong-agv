"""Wrap each missing code object as a tiny module .pyc and decompile with depyo."""
from __future__ import annotations

import importlib.util
import marshal
import struct
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYC_SRC = HERE.parents[0] / "__pycache__" / "solve_ecbs.cpython-312.pyc"
FN_DIR = HERE / "fn_pyc"
OUT_DIR = HERE / "fn_src"
FN_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

NEED = [
    "_build_overlap_delivery_timelines",
    "_embed_idle_patrol_motion",
    "_embed_spare_hub_motion",
    "_maybe_embed_pipeline_idles",
    "_bfs_cells",
    "_shortcut_path",
    "_paths_conflict",
    "_tighten_paths",
    "_spacetime_bfs",
    "_prioritized_st_paths",
    "_max_detour_ratio",
    "_bfs_len",
    "_chunk_list",
    "_wave_wait_ring",
    "_wave_wait_cells",
    "_wave_wait_cell",
    "_send_idles_wave_wait",
    "_pickup_load_agv",
    "_move_agvs_to_cells",
    "_delivery_stagger_cargo",
    "_pickup_stagger_turn_aware",
    "_pickup_stagger_paths",
    "_nearby_park",
    "_apply_paths",
    "_stitch_post_unload_stages",
    "_outer_leave_near_pick",
    "_leave_from_unload",
    "_outer_staging",
    "_apply_timelines_until_first_goal",
    "_apply_paths_until_first_goal",
    "_run_pipeline",
    "solve_ecbs",
    "main",
]


def wrap_code(code: types.CodeType, out: Path) -> None:
    stub = compile("def placeholder():\n    pass\n", "solve_ecbs.py", "exec")
    consts = list(stub.co_consts)
    for i, c in enumerate(consts):
        if isinstance(c, types.CodeType):
            consts[i] = code
            break
    mod = stub.replace(co_consts=tuple(consts), co_names=(code.co_name,))
    magic = importlib.util.MAGIC_NUMBER
    out.write_bytes(
        magic + struct.pack("<III", 0, int(time.time()), 0) + marshal.dumps(mod)
    )


def main() -> int:
    raw = PYC_SRC.read_bytes()
    mod = marshal.loads(raw[16:])
    found: dict[str, types.CodeType] = {}
    for c in mod.co_consts:
        if isinstance(c, types.CodeType) and c.co_name in NEED:
            found[c.co_name] = c
    print(f"found {len(found)}/{len(NEED)}", flush=True)

    py = sys.executable
    for name in NEED:
        code = found.get(name)
        if code is None:
            print(f"MISS {name}", flush=True)
            continue
        pyc = FN_DIR / f"{name}.pyc"
        wrap_code(code, pyc)
        src_path = OUT_DIR / f"{name}.py"
        proc = subprocess.run(
            [py, "-m", "depyo", "--out", str(pyc)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        text = proc.stdout or ""
        if not text.strip():
            text = (proc.stderr or "") + f"\n# exit={proc.returncode}\n"
        src_path.write_text(text, encoding="utf-8")
        bad = text.count("##ERROR##") + text.count(" = None\n")
        print(
            f"{name:40s} out={len(text):6d} err_marks~{bad} rc={proc.returncode}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
