# MioVerse AGV Map Editor

Interactive 20×20 grid editor (coords **1..20**) for redesigning warehouse scenarios **and** their transport task lists.

> **需要自定义场景宽高？** 用可变尺寸版：`python -m ml_research.map_editor_flex`（见 `ml_research/map_editor_flex/README.md`）。

## Launch

From repo root (`final_version`):

```bat
D:\condaEnvs\condaEnvs\agv-ml\python.exe -m ml_research.map_editor
```

## Two layers (important)

| Layer | Where | What you edit |
|-------|--------|----------------|
| **地图站台** | Left tools + grid | Obstacles, pickup/dropoff **stations**, AGV spawns (coordinates on the map) |
| **任务分配** | Right panel | Each task: **pickup station → dropoff destination** + **Urgent / Normal** |

Stations are places on the map; tasks assign which pickup goes to which dropoff (and urgency). Official L100 slots load both the map and that scene’s task CSV by default.

## Tools (map)

| Mode | Shortcut | Action |
|------|----------|--------|
| Obstacles | `O` | Click / drag paint; click again toggles |
| Pickup | `P` | Place pickup station (`start_point`) |
| Dropoff | `D` | Place dropoff station (`end_point`) |
| AGV | `A` | Place AGV; click same cell to cycle heading 0/90/180/270 |
| Erase | `E` | Clear cell |
| Right-click | — | Erase cell |
| Ctrl+Z | — | Undo |
| Ctrl+S | — | Save current slot |

Load **默认站台布局** for the standard 6 pickups + 16 dropoffs + 12 AGVs (keeps the current task list).

## Tasks panel

- **新建 / 编辑 / 删除** — CRUD on `task_id`, `start_point`, `end_point`, `priority` (`Normal`/`Urgent`), `remaining_time`
- **重载官方任务** — replace task list from that slot’s official L100 CSV (map unchanged)
- **清空任务** — clear assignments only
- Filter box to search by id / station / priority

## Slots

20 friendly slots: `SH_custom_01` … `SH_custom_20`.

- If a custom save exists → load custom map **and** custom tasks
- Else → load official SH01…SH20 **L100** map + that scene’s official task CSV

## Save locations

Saving writes to **both**:

1. `ml_research/map_editor/exports/SH_custom_XX.json` (+ `_position.csv`, `_task.csv`, `_obstacles.json`)
2. `ml_research/benchmarks/curriculum_shape/custom_maps/SH_custom_XX.json` (same set)

## Export format (JSON)

```json
{
  "id": "SH_custom_01",
  "format_version": 2,
  "grid_min": 1,
  "grid_max": 20,
  "pickups": [{"name": "Tiger", "x": 1, "y": 6}],
  "dropoffs": [{"name": "Beijing", "x": 6, "y": 4}],
  "agvs": [{"name": "Optimus", "x": 3, "y": 1, "pitch": 90}],
  "tasks": [
    {
      "task_id": "Tiger-1",
      "start_point": "Tiger",
      "end_point": "Suzhou",
      "priority": "Normal",
      "remaining_time": null
    }
  ],
  "extra_obstacles": [[5, 5], [5, 6]]
}
```

Companion **`_position.csv`**:

```text
type,name,x,y,pitch
start_point,Tiger,1,6,
end_point,Beijing,6,4,
agv,Optimus,3,1,90
```

Companion **`_task.csv`** (same columns as curriculum_shape ladder scenarios):

```text
task_id,start_point,end_point,priority,remaining_time
Tiger-1,Tiger,Suzhou,Normal,None
Rabbit-1,Rabbit,Dalian,Urgent,80
```

`extra_obstacles` is the same field used by curriculum_shape scenario JSON / runtime monkey-patch. Wire a slot into the ladder by pointing `position_csv` / `task_csv` at the exported files and copying `extra_obstacles` — without changing `main copy.py` A*.

## Notes

- Editable playable cells are **1..20** (y grows upward on screen, like `map_overview.py`).
- Colors match curriculum overview: blue pickup, green dropoff, dark obstacle, amber AGV; urgent tasks are highlighted in the list.
- Does not modify `main copy.py`, `shape_maps.py`, or lane_traffic_ai.
