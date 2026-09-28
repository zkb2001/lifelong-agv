"""Event-driven LLM coordinator for conflict-horizon MAPD.

Default policy remains max-effort A*. The LLM only returns keeper / yielder
roles + TTL. Paths stay in A* / yield parking. Backend is local Ollama.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

VALID_ACTS = ("PUSH", "YIELD_PARK", "EVACUATE", "HOLD_GATE", "RESUME")
DEFAULT_MODEL = os.environ.get("AGV_OLLAMA_MODEL", "deepseek-r1:latest")
DEFAULT_HOST = os.environ.get("AGV_OLLAMA_HOST", "http://127.0.0.1:11434")

SYSTEM = """你是仓库迷宫 MAPD 的高层协调员，不是路径规划器。
底层已经用 A* 按「最大努力」朝各自目标走。你只决定冲突时谁占用通道、谁让开、让开多久。禁止输出坐标路径或逐步动作。

## 场景（SH03 类）
- 网格 20×20，1-indexed，四连通。通道经常只有 1 格宽，无法侧向错车。
- 取货点在左右边墙：x=1（Tiger/Dragon/Horse）与 x=20（Rabbit/Ox/Monkey）。
- 卸货点在内部巷道（x≈6/9/12/15 的城市名站点）。
- 8 台 AGV；同时最多 max_active 台有任务。无任务车会去 staging，不能整局静止。
- 成功：deadline 内完成全部任务，且无 vertex 碰撞、无对穿 swap。

## 任务状态
assign → leg=to_pickup（空车去取货）→ 到达 pickup 后 loaded=true、leg=to_drop → 到达卸货点 done。
每车同时最多 1 单。idle = 当前无任务。

## 运动（你不要规划，但决策时要懂代价）
每秒只能 wait / turn / move 之一。转向 1s + 前进 1s。禁止穿墙、同格、对穿。

## 冲突类型与动作
- vertex：两车将占用同一格。让路车 YIELD_PARK：离开冲突格/走廊，尽量去 remA* 更小的侧向停车格。
- swap：两车对向互换格子（贴脸）。窄廊无法侧让。让路车必须 EVACUATE：沿来路退出对方占用带，允许本步 remA* 变大。禁止空 wait。
- 只选 1 个 keeper，动作 PUSH（继续原 A*）。其余全部让路。ttl 6–14 仿真秒。

## 选 keeper 的规则（按此顺序）
1. 有任务的车优先于 idle。idle 即使 remA*=0 也不能当 keeper（除非对方也是 idle）。
2. Urgent > 普通。
3. loaded（正在送货）> 空车去取货。
4. 有任务且 remA*≤12（快到取/卸点）应 PUSH，但 swap/贴脸除外：贴脸看谁清廊代价更小。
5. 其余比 remA* 小者优先。
6. 同一对反复冲突（pair_hits 高）时保持上次 keeper，不要互换，否则会乒乓死锁。

## 输出
只输出一个 JSON 对象，不要 markdown、不要分析、不要 <think>。
{"keeper":"<AGV名>","yielders":["<其他>"],"roles":{"<AGV>":{"act":"PUSH|YIELD_PARK|EVACUATE","ttl":8}},"reason":"<不超过20字>"}
keeper 必须是 involved 里的准确车名。swap/t_star=1 时让路 act 必须是 EVACUATE。

示例1 vertex，Jazz idle rem=0，Bumblebee loaded rem=18
{"keeper":"Bumblebee","yielders":["Jazz"],"roles":{"Bumblebee":{"act":"PUSH","ttl":10},"Jazz":{"act":"YIELD_PARK","ttl":8}},"reason":"送货车优先于idle"}

示例2 swap t_star=1，Hound loaded rem=26 vs Sideswipe empty rem=40
{"keeper":"Hound","yielders":["Sideswipe"],"roles":{"Hound":{"act":"PUSH","ttl":10},"Sideswipe":{"act":"EVACUATE","ttl":10}},"reason":"对向贴脸，空车倒退出廊"}

硬约束：回复的第一个字符必须是 `{`，最后一个字符必须是 `}`。不要写推理过程。
"""


@dataclass
class Role:
    act: str
    ttl: int = 8


@dataclass
class Decision:
    keeper: str
    yielders: List[str]
    roles: Dict[str, Role] = field(default_factory=dict)
    reason: str = ""
    source: str = "heuristic"


def _is_working_agent(a: dict) -> bool:
    if not a:
        return False
    if a.get("active") is False:
        return False
    leg = str(a.get("leg") or "")
    return leg in ("to_pickup", "to_drop") or bool(a.get("loaded"))


def _agent_by_name(snapshot: dict, name: str) -> dict:
    for a in snapshot.get("agents") or []:
        if str(a.get("name")) == name:
            return a
    return {}


def _heuristic(snapshot: dict, ranked: Sequence[str]) -> Decision:
    involved = list(snapshot.get("involved") or ranked)
    workers = [n for n in ranked if _is_working_agent(_agent_by_name(snapshot, n))]
    idlers = [n for n in ranked if n not in set(workers)]
    if workers:
        keeper = workers[0]
        yielders = [n for n in workers[1:] + idlers if n != keeper]
    else:
        keeper = ranked[0]
        yielders = list(ranked[1:])
    kind = str(snapshot.get("kind") or "")
    t_star = int(snapshot.get("t_star") or 99)
    evac = kind == "swap" or t_star <= 1
    roles = {keeper: Role("PUSH", 10)}
    for y in yielders:
        roles[y] = Role("EVACUATE" if (evac or not _is_working_agent(_agent_by_name(snapshot, y))) else "YIELD_PARK", 8)
    return Decision(
        keeper=keeper,
        yielders=yielders,
        roles=roles,
        reason="heuristic worker>urgent>loaded>remA*",
        source="heuristic",
    )


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```(?:think|thinking).*?```", "", text, flags=re.S | re.I)
    return text.strip()


def _fallback_from_prose(text: str, involved: Sequence[str]) -> Optional[dict]:
    names = set(involved)
    patterns = (
        r"(\w+) should be (?:the )?keeper",
        r"keeper (?:is|should be) (\w+)",
        r"choose (\w+) as (?:the )?keeper",
        r"(\w+) (?:as|for) keeper",
        r"keeper[\"']?\s*[:=]\s*[\"']?(\w+)",
        r"让(\w+)(?:继续|占用|PUSH|通过)",
        r"(?:选择|应选|选)\s*(\w+)\s*(?:作为|当|为)?\s*keeper",
        r"(\w+)\s*(?:作为|当)\s*keeper",
        r"keeper[=：:]\s*(\w+)",
        r"\"keeper\"\s*:\s*\"(\w+)\"",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            cand = m.group(1)
            if cand in names:
                others = [n for n in involved if n != cand]
                return {
                    "keeper": cand,
                    "yielders": others,
                    "roles": {},
                    "reason": "prose_fallback",
                }
    return None


def _extract_json(text: str) -> Optional[dict]:
    text = _strip_think(text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "keeper" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    for m in reversed(list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.S))):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "keeper" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _validate(obj: dict, involved: Sequence[str], fallback: Decision, snapshot: Optional[dict] = None) -> Decision:
    names = set(involved)
    keeper = str(obj.get("keeper") or "")
    if keeper not in names:
        return fallback
    rem: Dict[str, int] = {}
    working: Dict[str, bool] = {}
    for a in snapshot.get("agents") or [] if snapshot else []:
        n = str(a.get("name"))
        try:
            rem[n] = int(a.get("rem_astar") or 10**6)
        except (TypeError, ValueError):
            rem[n] = 10**6
        working[n] = _is_working_agent(a)
    kind = str((snapshot or {}).get("kind") or "")
    t_star = int((snapshot or {}).get("t_star") or 99)
    # remA*≤12 只保护「快完成的任务车」，不要把停在 staging 的 idle(rem=0) 抬成 keeper
    near = [
        n
        for n in involved
        if working.get(n) and rem.get(n, 10**6) <= 12
    ]
    if near and kind != "swap" and t_star > 1:
        if not working.get(keeper):
            keeper = min(near, key=lambda n: (rem.get(n, 10**6), n))
        elif rem.get(keeper, 10**6) > 12:
            keeper = min(near, key=lambda n: (rem.get(n, 10**6), n))
    raw_y = obj.get("yielders")
    if isinstance(raw_y, str):
        raw_y = [raw_y]
    yielders = [str(y) for y in (raw_y or []) if y in names and y != keeper]
    for n in involved:
        if n != keeper and n not in yielders:
            yielders.append(n)
    roles: Dict[str, Role] = {}
    raw_roles = obj.get("roles") if isinstance(obj.get("roles"), dict) else {}
    evac_needed = kind == "swap" or t_star <= 1
    for n in involved:
        rr = raw_roles.get(n) if isinstance(raw_roles, dict) else None
        act = "PUSH" if n == keeper else ("EVACUATE" if evac_needed else "YIELD_PARK")
        ttl = 8
        if isinstance(rr, dict):
            cand = str(rr.get("act") or act).upper()
            if cand in VALID_ACTS:
                if n == keeper and cand in ("YIELD_PARK", "EVACUATE", "HOLD_GATE"):
                    act = "PUSH"
                elif n != keeper and cand == "PUSH":
                    act = "EVACUATE" if evac_needed else "YIELD_PARK"
                else:
                    act = cand
            try:
                ttl = max(3, min(24, int(rr.get("ttl", ttl))))
            except (TypeError, ValueError):
                ttl = 8
        roles[n] = Role(act, ttl)
    return Decision(
        keeper=keeper,
        yielders=yielders,
        roles=roles,
        reason=str(obj.get("reason") or "")[:200],
        source="ollama",
    )


def _is_deepseek(model: str) -> bool:
    return "deepseek" in model.lower()


def _fmt_agent_line(a: dict) -> str:
    name = a.get("name")
    pos = a.get("pos")
    pitch = a.get("pitch")
    leg = a.get("leg") or ("idle" if not a.get("active") else "?")
    tid = a.get("task_id") or "-"
    dest = a.get("destination") or "-"
    goal = a.get("goal")
    path = a.get("path_next") or []
    flags = []
    if a.get("loaded"):
        flags.append("loaded")
    if a.get("urgent"):
        flags.append("URGENT")
    if a.get("yield_locked"):
        flags.append("yield_locked")
    if not a.get("active"):
        flags.append("idle")
    flag_s = ",".join(flags) if flags else "empty-working"
    return (
        f"- {name} xy={pos} pitch={pitch} {flag_s} leg={leg} "
        f"task={tid} dest={dest} goal={goal} remA*={a.get('rem_astar')} "
        f"path={path}"
    )


def build_user_prompt(snapshot: dict, ranked: Sequence[str], pair_hits: int) -> str:
    involved = list(snapshot.get("involved") or ranked)
    kind = snapshot.get("kind")
    t_star = snapshot.get("t_star")
    scene = str(snapshot.get("scene") or "").strip()
    agents = list(snapshot.get("agents") or [])
    by = {str(a.get("name")): a for a in agents}
    involved_lines = [
        _fmt_agent_line(by[n]) for n in involved if n in by
    ]
    others = [a for a in agents if str(a.get("name")) not in set(involved)]
    fleet_lines = [_fmt_agent_line(a) for a in others]
    suggested = ranked[0] if ranked else ""
    evac_hint = (
        "本冲突是 swap 或贴脸：让路车必须 EVACUATE，不要找侧向更近停车格。"
        if kind == "swap" or int(snapshot.get("t_star") or 99) <= 1
        else "本冲突是 vertex：让路车 YIELD_PARK 离开冲突格即可。"
    )
    parts = [
        scene,
        "",
        f"## 当前冲突 sim_t={snapshot.get('t')} deadline={snapshot.get('deadline')} "
        f"完成={snapshot.get('done')}/{snapshot.get('total')} wave目标=清掉本对冲突后继续最大努力",
        f"类型={kind} t_star={t_star} pair_hits={pair_hits} "
        f"same_pose={snapshot.get('same_pose_streak', 0)}",
        f"冲突双方: {' vs '.join(involved)}",
        evac_hint,
        "规则建议 keeper（可推翻，但请说明）: " + suggested,
        "",
        "## 冲突双方（含任务与接下来的 A* 格子）",
        *involved_lines,
    ]
    if fleet_lines:
        parts += ["", "## 其余 AGV（避免把通道让给会堵住全图的车）", *fleet_lines]
    parts += [
        "",
        "选出 1 个 keeper。下面请直接给出 JSON（第一个字符是 {）：",
    ]
    return "\n".join(p for p in parts if p is not None)


def ollama_chat(
    messages: List[dict],
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 90.0,
) -> str:
    url = host.rstrip("/") + "/api/chat"

    def _post(body: dict) -> dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # Prefill `{` so reasoning models cannot start with an essay.
    msgs = list(messages)
    if not msgs or msgs[-1].get("role") != "assistant":
        msgs = msgs + [{"role": "assistant", "content": "{"}]
    predict = 256
    base: Dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": predict},
        "format": "json",
        "think": False,
    }
    try:
        raw = _post(base)
    except urllib.error.HTTPError:
        body = dict(base)
        body.pop("format", None)
        try:
            raw = _post(body)
        except urllib.error.HTTPError:
            body.pop("think", None)
            raw = _post(body)
    msg = raw.get("message") or {}
    content = str(msg.get("content") or "").strip()
    thinking = str(msg.get("thinking") or "").strip()
    text = content or thinking
    if text and not text.lstrip().startswith("{"):
        text = "{" + text
    return text


class LlmCoordinator:
    def __init__(
        self,
        *,
        enabled: bool = True,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        cooldown: int = 8,
        min_hits: int = 1,
        timeout: float = 90.0,
        scene: str = "",
    ) -> None:
        self.enabled = enabled
        self.model = model
        self.host = host
        self.cooldown = cooldown
        self.min_hits = min_hits
        self.timeout = timeout
        self.scene = scene
        self.last_call_t = -10**9
        self.n_calls = 0
        self.n_ok = 0
        self.n_fail = 0
        self.n_skip = 0
        self.log: List[dict] = []

    def decide(
        self,
        snapshot: dict,
        ranked: Sequence[str],
        *,
        now: int,
        pair_hits: int,
    ) -> Decision:
        if self.scene and not snapshot.get("scene"):
            snapshot = {**snapshot, "scene": self.scene}
        fb = _heuristic(snapshot, ranked)
        if not self.enabled:
            self.n_skip += 1
            return fb
        if snapshot.get("skip_llm"):
            self.n_skip += 1
            fb.source = "heuristic_skip_llm"
            return fb
        if pair_hits < self.min_hits:
            self.n_skip += 1
            fb.source = "heuristic_early"
            return fb
        if now - self.last_call_t < self.cooldown:
            self.n_skip += 1
            fb.source = "heuristic_cooldown"
            return fb
        self.last_call_t = now
        self.n_calls += 1
        user = build_user_prompt(snapshot, ranked, pair_hits)
        try:
            text = ollama_chat(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                model=self.model,
                host=self.host,
                timeout=self.timeout,
            )
            obj = _extract_json(text)
            if not obj:
                obj = _fallback_from_prose(
                    text, list(snapshot.get("involved") or ranked)
                )
            if not obj:
                raise ValueError("no json")
            dec = _validate(obj, list(snapshot.get("involved") or ranked), fb, snapshot)
            if dec.source != "ollama":
                raise ValueError("invalid keeper")
            self.n_ok += 1
            self.log.append(
                {
                    "t": now,
                    "keeper": dec.keeper,
                    "yielders": dec.yielders,
                    "reason": dec.reason,
                    "source": dec.source,
                    "raw": text[:400],
                }
            )
            return dec
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as e:
            self.n_fail += 1
            fb.source = f"heuristic_fail:{type(e).__name__}"
            fb.reason = str(e)[:120]
            self.log.append({"t": now, "source": fb.source, "err": fb.reason})
            return fb

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "calls": self.n_calls,
            "ok": self.n_ok,
            "fail": self.n_fail,
            "skip": self.n_skip,
        }
