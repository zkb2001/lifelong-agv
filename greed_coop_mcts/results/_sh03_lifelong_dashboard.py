"""SH03 lifelong live dashboard: map + AGV poses + progress charts.

Serves http://127.0.0.1:8765/ — polls SH03_gcm_live.json + map exports.
"""
from __future__ import annotations

import csv
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from greed_coop_mcts.paths import MAP_EXPORTS, RESULTS

HOST = "127.0.0.1"
PORT = 8765
# solve() sid comes from map meta id (SH_custom_03), not SH03_gcm.
LIVE = RESULTS / "SH_custom_03_live.json"
LIVE_FALLBACKS = (
    RESULTS / "SH03_gcm_live.json",
    RESULTS / "SH03_live.json",
)
LOG = RESULTS / "sh03_lifelong.log"
MAP_JSON = MAP_EXPORTS / "SH_custom_03.json"
POS_CSV = MAP_EXPORTS / "SH_custom_03_position.csv"
OBS_JSON = MAP_EXPORTS / "SH_custom_03_obstacles.json"

_PROG_RE = re.compile(
    r"\[(?P<sid>[^\]]+)/gcm\]\s+t=(?P<t>\d+)\s+done=(?P<done>\d+)/(?P<total>\d+)."
    r"*?wall=(?P<wall>[\d.]+)s"
)

_cache: Dict[str, Any] = {
    "history": [],  # {t, done, total, wall, k}
    "log_pos": 0,
    "map": None,
}


def _load_map() -> dict:
    w = h = 20
    obstacles: List[Tuple[int, int]] = []
    pickups: Dict[str, Tuple[int, int]] = {}
    dropoffs: Dict[str, Tuple[int, int]] = {}
    agv0: Dict[str, dict] = {}

    def _ingest_obs(cells) -> None:
        for o in cells or []:
            if isinstance(o, dict):
                obstacles.append((int(o["x"]), int(o["y"])))
            elif isinstance(o, (list, tuple)) and len(o) >= 2:
                obstacles.append((int(o[0]), int(o[1])))

    if MAP_JSON.is_file():
        j = json.loads(MAP_JSON.read_text(encoding="utf-8"))
        gmin = int(j.get("grid_min") or 1)
        gmax = int(j.get("grid_max") or 20)
        w = int(j.get("width") or j.get("W") or (gmax - gmin + 1))
        h = int(j.get("height") or j.get("H") or (gmax - gmin + 1))
        for p in j.get("pickups") or []:
            pickups[str(p["name"])] = (int(p["x"]), int(p["y"]))
        for p in j.get("dropoffs") or []:
            dropoffs[str(p["name"])] = (int(p["x"]), int(p["y"]))
        _ingest_obs(j.get("obstacles"))
        _ingest_obs(j.get("extra_obstacles"))

    if POS_CSV.is_file():
        with POS_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = (row.get("type") or "").strip()
                name = (row.get("name") or "").strip()
                try:
                    x, y = int(row["x"]), int(row["y"])
                except (KeyError, ValueError):
                    continue
                if t == "start_point":
                    pickups.setdefault(name, (x, y))
                elif t == "end_point":
                    dropoffs.setdefault(name, (x, y))
                elif t == "agv":
                    agv0[name] = {
                        "name": name,
                        "x": x,
                        "y": y,
                        "pitch": int(float(row.get("pitch") or 0)) % 360,
                    }

    if OBS_JSON.is_file():
        raw = json.loads(OBS_JSON.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            _ingest_obs(raw)
        else:
            _ingest_obs(raw.get("obstacles"))
            _ingest_obs(raw.get("extra_obstacles"))

    return {
        "width": w,
        "height": h,
        "obstacles": obstacles,
        "pickups": {k: {"x": v[0], "y": v[1]} for k, v in pickups.items()},
        "dropoffs": {k: {"x": v[0], "y": v[1]} for k, v in dropoffs.items()},
        "agv_home": list(agv0.values()),
    }


def _tail_log_history() -> None:
    if not LOG.is_file():
        return
    try:
        size = LOG.stat().st_size
    except OSError:
        return
    pos = int(_cache["log_pos"])
    if size < pos:
        pos = 0
        _cache["history"] = []
    with LOG.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(pos)
        chunk = f.read()
        _cache["log_pos"] = f.tell()
    hist: List[dict] = _cache["history"]
    for line in chunk.splitlines():
        m = _PROG_RE.search(line)
        if not m:
            continue
        hist.append(
            {
                "t": int(m.group("t")),
                "done": int(m.group("done")),
                "total": int(m.group("total")),
                "wall": float(m.group("wall")),
            }
        )
    if len(hist) > 2000:
        _cache["history"] = hist[-2000:]


def _live_path() -> Optional[Path]:
    for p in (LIVE, *LIVE_FALLBACKS):
        if p.is_file():
            return p
    cands = sorted(
        RESULTS.glob("*_live.json"), key=lambda x: x.stat().st_mtime, reverse=True
    )
    return cands[0] if cands else None


def _read_live() -> Optional[dict]:
    path = _live_path()
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _status_payload() -> dict:
    _tail_log_history()
    if _cache["map"] is None:
        _cache["map"] = _load_map()
    live = _read_live()
    hist = list(_cache["history"])
    if live:
        pt = {
            "t": int(live.get("t") or 0),
            "done": int(live.get("done") or 0),
            "total": int(live.get("total") or 0),
            "wall": float(live.get("wall_s") or 0),
            "k": live.get("k"),
        }
        if not hist or hist[-1].get("t") != pt["t"] or hist[-1].get("done") != pt["done"]:
            hist = hist + [pt]
            _cache["history"] = hist[-2000:]
    running = False
    if live and live.get("updated_at"):
        # consider live if file mtime recent
        try:
            age = time.time() - LIVE.stat().st_mtime
            running = age < 90
        except OSError:
            running = False
    return {
        "live": live,
        "history": hist,
        "map": _cache["map"],
        "running": running,
        "log_exists": LOG.is_file(),
        "live_path": str(LIVE),
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SH03 Lifelong</title>
<style>
  :root {
    --bg0: #0e1419;
    --bg1: #152028;
    --line: #243440;
    --ink: #e7eef3;
    --muted: #8aa0ad;
    --accent: #3db8a0;
    --warn: #e0a35a;
    --pickup: #4ea1ff;
    --drop: #e07a5f;
    --agv: #f4d35e;
    --agv-load: #f07167;
    --obs: #1c2a33;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1a3038 0%, var(--bg0) 55%);
    color: var(--ink); min-height: 100vh;
  }
  header {
    display: flex; align-items: baseline; gap: 1rem; padding: 1rem 1.5rem 0.5rem;
    border-bottom: 1px solid var(--line);
  }
  header h1 { margin: 0; font-size: 1.35rem; letter-spacing: 0.02em; font-weight: 650; }
  header .sub { color: var(--muted); font-size: 0.9rem; }
  .pill {
    margin-left: auto; padding: 0.25rem 0.7rem; border-radius: 999px;
    font-size: 0.8rem; border: 1px solid var(--line); color: var(--muted);
  }
  .pill.on { color: #0b1a16; background: var(--accent); border-color: transparent; font-weight: 600; }
  main {
    display: grid;
    grid-template-columns: minmax(320px, 1.1fr) minmax(280px, 0.9fr);
    gap: 1rem; padding: 1rem 1.5rem 1.5rem;
  }
  @media (max-width: 960px) { main { grid-template-columns: 1fr; } }
  .panel {
    background: color-mix(in srgb, var(--bg1) 92%, #000);
    border: 1px solid var(--line); border-radius: 12px; padding: 0.9rem 1rem;
  }
  .panel h2 { margin: 0 0 0.75rem; font-size: 0.95rem; color: var(--muted); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
  .metrics {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-bottom: 0.9rem;
  }
  .metric {
    background: var(--bg0); border: 1px solid var(--line); border-radius: 10px; padding: 0.65rem 0.75rem;
  }
  .metric .lab { font-size: 0.72rem; color: var(--muted); }
  .metric .val { font-size: 1.25rem; font-weight: 650; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }
  .metric .val small { font-size: 0.75rem; color: var(--muted); font-weight: 500; }
  #mapWrap { position: relative; width: 100%; aspect-ratio: 1; background: #0a1014; border-radius: 10px; overflow: hidden; border: 1px solid var(--line); }
  #map { width: 100%; height: 100%; display: block; }
  #chart { width: 100%; height: 180px; display: block; background: var(--bg0); border-radius: 10px; border: 1px solid var(--line); }
  .legend { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.6rem; font-size: 0.78rem; color: var(--muted); }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 0.35rem; vertical-align: -1px; }
  .bar {
    height: 8px; background: var(--bg0); border-radius: 999px; overflow: hidden; border: 1px solid var(--line); margin: 0.4rem 0 0.8rem;
  }
  .bar > span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), #6ed4c0); width: 0%; transition: width 0.4s ease; }
  .meta { font-size: 0.8rem; color: var(--muted); line-height: 1.55; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<header>
  <h1>MioVerse · SH03 Lifelong</h1>
  <span class="sub">seed 400 → random stream · FACN v1</span>
  <span id="runPill" class="pill">等待数据…</span>
</header>
<main>
  <section class="panel">
    <h2>地图实时</h2>
    <div id="mapWrap"><canvas id="map"></canvas></div>
    <div class="legend">
      <span><i style="background:var(--obs)"></i>障碍</span>
      <span><i style="background:var(--pickup)"></i>取货站</span>
      <span><i style="background:var(--drop)"></i>卸货点</span>
      <span><i style="background:var(--agv)"></i>空载 AGV</span>
      <span><i style="background:var(--agv-load)"></i>载货 AGV</span>
    </div>
  </section>
  <section class="panel">
    <h2>进度</h2>
    <div class="metrics">
      <div class="metric"><div class="lab">完成 / 目标</div><div class="val" id="mDone">—</div></div>
      <div class="metric"><div class="lab">仿真时 t</div><div class="val" id="mT">—</div></div>
      <div class="metric"><div class="lab">墙钟</div><div class="val" id="mWall">—</div></div>
      <div class="metric"><div class="lab">FACN K</div><div class="val" id="mK">—</div></div>
      <div class="metric"><div class="lab">在途 / 接近</div><div class="val" id="mBusy">—</div></div>
      <div class="metric"><div class="lab">队列待派</div><div class="val" id="mPend">—</div></div>
    </div>
    <div class="bar"><span id="progBar"></span></div>
    <canvas id="chart"></canvas>
    <div class="meta" id="meta" style="margin-top:0.75rem"></div>
  </section>
</main>
<script>
const CELL = {obs:'#1c2a33', pickup:'#4ea1ff', drop:'#e07a5f', agv:'#f4d35e', load:'#f07167', grid:'#18242c', ink:'#dfe8ee'};
let lastMap = null;

function drawMap(map, live) {
  const canvas = document.getElementById('map');
  const wrap = document.getElementById('mapWrap');
  const dpr = window.devicePixelRatio || 1;
  const css = Math.min(wrap.clientWidth, wrap.clientHeight);
  canvas.width = css * dpr; canvas.height = css * dpr;
  canvas.style.width = css + 'px'; canvas.style.height = css + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = map.width || 20, H = map.height || 20;
  const s = css / Math.max(W, H);
  ctx.fillStyle = '#0a1014'; ctx.fillRect(0,0,css,css);
  const obs = new Set((map.obstacles||[]).map(p => p[0]+','+p[1]));
  for (let y=1; y<=H; y++) for (let x=1; x<=W; x++) {
    const px=(x-1)*s, py=(H-y)*s;
    ctx.fillStyle = obs.has(x+','+y) ? CELL.obs : '#101820';
    ctx.fillRect(px+0.5, py+0.5, s-1, s-1);
  }
  const mark = (dict, color) => {
    Object.entries(dict||{}).forEach(([name, p]) => {
      const px=(p.x-1)*s, py=(H-p.y)*s;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.roundRect(px+s*0.15, py+s*0.15, s*0.7, s*0.7, 3); ctx.fill();
      if (s > 14) {
        ctx.fillStyle = '#0a1014'; ctx.font = Math.max(8, s*0.28)+'px sans-serif';
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText(name.slice(0,3), px+s/2, py+s/2);
      }
    });
  };
  mark(map.dropoffs, CELL.drop);
  mark(map.pickups, CELL.pickup);
  const agvs = (live && live.agvs) ? live.agvs : (map.agv_home||[]);
  agvs.forEach(a => {
    const px=(a.x-1)*s + s/2, py=(H-a.y)*s + s/2;
    ctx.fillStyle = a.loaded ? CELL.load : CELL.agv;
    ctx.beginPath(); ctx.arc(px, py, s*0.28, 0, Math.PI*2); ctx.fill();
    const rad = ((90 - (a.pitch||0)) * Math.PI) / 180;
    ctx.strokeStyle = '#0a1014'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, py);
    ctx.lineTo(px + Math.cos(rad)*s*0.32, py - Math.sin(rad)*s*0.32); ctx.stroke();
  });
}

function drawChart(hist) {
  const canvas = document.getElementById('chart');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = 180;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0a1014'; ctx.fillRect(0,0,w,h);
  if (!hist || hist.length < 2) {
    ctx.fillStyle = '#8aa0ad'; ctx.font = '13px sans-serif';
    ctx.fillText('等待进度点…', 16, h/2); return;
  }
  const pad = {l:40, r:12, t:12, b:28};
  const maxT = Math.max(...hist.map(p=>p.t||0), 1);
  const maxD = Math.max(...hist.map(p=>p.done||0), 1);
  const maxTot = Math.max(...hist.map(p=>p.total||0), maxD, 1);
  const xOf = t => pad.l + (t/maxT)*(w-pad.l-pad.r);
  const yOf = d => pad.t + (1 - d/maxTot)*(h-pad.t-pad.b);
  ctx.strokeStyle = '#243440'; ctx.lineWidth = 1;
  for (let i=0;i<=4;i++){
    const yy = pad.t + i*(h-pad.t-pad.b)/4;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w-pad.r, yy); ctx.stroke();
    ctx.fillStyle='#8aa0ad'; ctx.font='10px sans-serif'; ctx.textAlign='right';
    ctx.fillText(String(Math.round(maxTot*(1-i/4))), pad.l-6, yy+3);
  }
  // total offered
  ctx.strokeStyle = '#3a5160'; ctx.setLineDash([4,4]); ctx.beginPath();
  hist.forEach((p,i)=>{ const x=xOf(p.t), y=yOf(p.total); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.stroke(); ctx.setLineDash([]);
  // done
  ctx.strokeStyle = '#3db8a0'; ctx.lineWidth = 2; ctx.beginPath();
  hist.forEach((p,i)=>{ const x=xOf(p.t), y=yOf(p.done); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.stroke();
  ctx.fillStyle='#8aa0ad'; ctx.font='10px sans-serif'; ctx.textAlign='left';
  ctx.fillText('t →', pad.l, h-8);
  ctx.fillText('done (实) / offered (虚)', pad.l+28, h-8);
}

function fmt(n){ return (n==null||n===undefined)?'—':String(n); }
function tick(){
  fetch('/api/status').then(r=>r.json()).then(data=>{
    const live = data.live;
    const map = data.map; lastMap = map;
    document.getElementById('runPill').textContent = data.running ? 'RUNNING' : (live ? 'IDLE / 结束' : '等待启动');
    document.getElementById('runPill').className = 'pill' + (data.running ? ' on' : '');
    if (live) {
      const tot = live.lifelong_max_total || live.total || 800;
      document.getElementById('mDone').innerHTML = `${live.done}<small> / ${tot}</small>`;
      document.getElementById('mT').textContent = live.t;
      document.getElementById('mWall').innerHTML = `${live.wall_s}<small>s</small>`;
      const kd = live.facn_delta!=null ? ` Δ${live.facn_delta}` : '';
      document.getElementById('mK').innerHTML = `${fmt(live.k)}<small>${kd}</small>`;
      document.getElementById('mBusy').textContent = `${live.carry} / ${live.approach}`;
      document.getElementById('mPend').textContent = live.pending;
      const pct = Math.min(100, 100 * (live.done / Math.max(1, tot)));
      document.getElementById('progBar').style.width = pct + '%';
      const phase = live.stream_on ? '随机流' : '种子段';
      document.getElementById('meta').innerHTML =
        `阶段: <b style="color:#e7eef3">${phase}</b> · 种子 ${live.initial_total} · 已流式 ${live.n_streamed||0}<br/>` +
        `更新: ${live.updated_at || '—'} · stall=${live.stall} · server ${data.server_time}`;
    }
    drawMap(map, live);
    drawChart(data.history || []);
  }).catch(()=>{});
}
tick(); setInterval(tick, 1500);
window.addEventListener('resize', ()=>{ if(lastMap) drawMap(lastMap, null); });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            payload = json.dumps(_status_payload(), ensure_ascii=False).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    _cache["map"] = _load_map()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"SH03 lifelong dashboard → http://{HOST}:{PORT}/", flush=True)
    print(f"  live={LIVE}", flush=True)
    print(f"  log ={LOG}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
