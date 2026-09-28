from pathlib import Path
import ast

p = Path(r"d:\compitition\MioVerse\final_version\ml_research\benchmarks\coord_custom_ai\solve_ecbs.py")
text = p.read_text(encoding="utf-8")

old = """                                    _clear_after_unload(agv)
                        if _step_off_buf:
                            now = _fleet_hold_tick(
                                steps_by,
                                pose,
                                names,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                            )
                        _flush_step_off()
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f\"[ECBS] delivery ST leftover-parallel \"
                                f\"k={len(ok_need)} left={len(need)}\",
                                flush=True,
                            )"""

new = """                                    _clear_after_unload(agv)
                            if _step_off_buf:
                                now = _fleet_hold_tick(
                                    steps_by,
                                    pose,
                                    names,
                                    now,
                                    loaded=loaded,
                                    dest=dest,
                                    tid=tid,
                                )
                            _flush_step_off()
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f\"[ECBS] delivery ST leftover-parallel \"
                                f\"k={len(ok_need)} left={len(need)}\",
                                flush=True,
                            )"""

# Also fix any B-pattern mis-indents: look for "_flush_step_off()\n                            need"
bad = "                        _flush_step_off()\n                            need = sorted("
good = "                            _flush_step_off()\n                            need = sorted("

n = 0
if old in text:
    text = text.replace(old, new, 1)
    n += 1
    print("fixed leftover-parallel block")

# generic: hold block at wrong indent before need=
import re
# Fix pattern where hold is at 24 spaces then need at 28
pat = re.compile(
    r"\n(?P<hold>                        if _step_off_buf:\n"
    r"                            now = _fleet_hold_tick\(\n"
    r"                                steps_by,\n"
    r"                                pose,\n"
    r"                                names,\n"
    r"                                now,\n"
    r"                                loaded=loaded,\n"
    r"                                dest=dest,\n"
    r"                                tid=tid,\n"
    r"                            \)\n"
    r"                        _flush_step_off\(\)\n)"
    r"(?P<need>                            (?:need|print))"
)
def repl(m):
    hold = m.group("hold").replace("\n                        ", "\n                            ").replace(
        "\n                            now", "\n                                now"
    ).replace(
        "\n                                steps_by", "\n                                    steps_by"
    ).replace(
        "\n                                pose", "\n                                    pose"
    ).replace(
        "\n                                names", "\n                                    names"
    ).replace(
        "\n                                now,", "\n                                    now,"
    ).replace(
        "\n                                loaded", "\n                                    loaded"
    ).replace(
        "\n                                dest", "\n                                    dest"
    ).replace(
        "\n                                tid", "\n                                    tid"
    ).replace(
        "\n                            )", "\n                                )"
    ).replace(
        "\n                        _flush", "\n                            _flush"
    )
    # simpler approach below
    return m.group(0)

# Simpler: count remaining bad patterns and fix with direct replace
while bad in text:
    text = text.replace(bad, good, 1)
    n += 1
    print("fixed bad flush/need indent")

# Also fix hold-if at 24 spaces when followed by need at 28
bad2 = """                        if _step_off_buf:
                            now = _fleet_hold_tick(
                                steps_by,
                                pose,
                                names,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                            )
                        _flush_step_off()
                            print("""
# too specific

p.write_text(text, encoding="utf-8")
try:
    ast.parse(text)
    print("AST ok, patches", n)
except SyntaxError as e:
    print("AST FAIL", e)
    # show context
    lines = text.splitlines()
    ln = e.lineno or 1
    for i in range(max(0, ln - 8), min(len(lines), ln + 8)):
        print(f"{i+1}:{lines[i]}")
