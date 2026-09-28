from pathlib import Path

p = Path(__file__).with_name("solve_ecbs.py")
t = p.read_text(encoding="utf-8")
print("bytes", p.stat().st_size)
print("lines", t.count("\n") + 1)
print("hold", t.count("_fleet_hold_tick"))
print("turn", t.count("_turn_pitches_toward"))
print("bfs", "Do NOT BFS-snap" in t)
print("FALSE loaded", t.count('[-1]["loaded"] = "FALSE"'))
print("waves_ecbs", "waves_ecbs" in t or "before-claim" in t)
# show snippets around remaining FALSE flips
idx = 0
n = 0
needle = '[-1]["loaded"] = "FALSE"'
while True:
    i = t.find(needle, idx)
    if i < 0:
        break
    n += 1
    line = t[:i].count("\n") + 1
    ctx = t[max(0, i - 200) : i + 120].replace("\n", "\\n")
    print(f"--- flip {n} @L{line} ---")
    print(ctx[:300])
    idx = i + len(needle)
print("total flips", n)
