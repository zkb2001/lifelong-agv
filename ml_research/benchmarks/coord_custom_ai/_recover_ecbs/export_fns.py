"""Export per-function marshal blobs from Sep-2 solve_ecbs.cpython-312.pyc."""
from __future__ import annotations

import importlib.util
import marshal
import types
from pathlib import Path

pyc_path = Path(__file__).resolve().parents[1] / "__pycache__" / "solve_ecbs.cpython-312.pyc"
out_dir = Path(__file__).resolve().parent / "fn_pyc"
out_dir.mkdir(exist_ok=True)

raw = pyc_path.read_bytes()
mod = marshal.loads(raw[16:])
need = {
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
}
found: dict[str, types.CodeType] = {}
for c in mod.co_consts:
    if isinstance(c, types.CodeType) and c.co_name in need:
        found[c.co_name] = c

print("found", len(found), "/", len(need))
magic = importlib.util.MAGIC_NUMBER
# pyc header: magic + flags + hash/timestamp + size (16 bytes on 3.7+)
import struct
import time

flags = 0
mtime = int(time.time())
for name, code in found.items():
    data = magic + struct.pack("<I", flags) + struct.pack("<II", mtime, 0) + marshal.dumps(code)
    # Note: dumping a lone code object as module pyc is invalid for import,
    # but depyo --marshal can take raw marshal of a code object.
    (out_dir / f"{name}.marshal").write_bytes(marshal.dumps(code))
    # Also wrap as a fake module containing only this function for better decomp.
    # Build a synthetic module code object is hard; skip.
print("wrote", out_dir)
