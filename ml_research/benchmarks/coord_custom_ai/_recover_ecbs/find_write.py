import json
from pathlib import Path

root = Path(
    r"C:/Users/zkb17/.cursor/projects/d-compitition-MioVerse-final-version/agent-transcripts"
)
out = Path(
    r"d:/compitition/MioVerse/final_version/ml_research/benchmarks/coord_custom_ai/_recover_ecbs"
)
out.mkdir(parents=True, exist_ok=True)
best = None  # (len, path, contents)
n = 0
for p in root.rglob("*.jsonl"):
    for line in p.open(encoding="utf-8", errors="replace"):
        if "solve_ecbs.py" not in line or "Write" not in line:
            continue
        if "contents" not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        for part in obj.get("message", {}).get("content", []) or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "tool_use" or part.get("name") != "Write":
                continue
            inp = part.get("input") or {}
            path = str(inp.get("path") or "")
            if "solve_ecbs.py" not in path.replace("\\", "/"):
                continue
            c = inp.get("contents")
            if not c or "def solve_ecbs" not in c:
                continue
            n += 1
            if best is None or len(c) > best[0]:
                best = (len(c), str(p), c)

print("writes_with_solve", n)
if best:
    print("best_len", best[0], "from", best[1])
    (out / "solve_ecbs_from_transcript.py").write_text(best[2], encoding="utf-8")
    print("wrote", out / "solve_ecbs_from_transcript.py")
else:
    print("none")
