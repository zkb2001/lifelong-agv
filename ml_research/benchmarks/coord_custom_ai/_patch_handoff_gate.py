"""Wire m0_handoff_enabled / AGV_M0_HANDOFF into allow_yield."""
from pathlib import Path
import ast

p = Path(__file__).with_name("solve_ecbs.py")
text = p.read_text(encoding="utf-8")

# 1) Inside _solve_via_m0_engine after traffic_recovery block
old1 = '''    # Rule recovery replaces classifier→ECBS escalate
    if traffic_recovery:
        allow_yield = False
        gate = None

    print(
        f"[M0] engine"'''
# encoding may have special arrow - use looser match
if old1 not in text:
    old1 = '''    if traffic_recovery:
        allow_yield = False
        gate = None

    print(
        f"[M0] engine"'''

new1 = '''    if traffic_recovery:
        allow_yield = False
        gate = None

    # Opt-out: AGV_M0_HANDOFF=0 or meta.m0_handoff_enabled=False keeps pure M0.
    _hf = str(_os.environ.get("AGV_M0_HANDOFF", "1")).strip().lower()
    if _hf in ("0", "false", "no", "off"):
        allow_yield = False
        gate = None
    if meta.get("m0_handoff_enabled") is False:
        allow_yield = False
        gate = None

    print(
        f"[M0] engine"'''

if text.count(old1) != 1:
    raise SystemExit(f"anchor1 count={text.count(old1)}")
text = text.replace(old1, new1, 1)
print("ok engine gate")

# 2) Call site: respect meta / env before allow_yield=bool(allow_switch)
old2 = '''                gate=gate if allow_switch else None,
                allow_yield=bool(allow_switch),
                traffic_recovery=False,
            )
            if not m0_rep.get("handoff"):
                return m0_rep'''
new2 = '''                gate=gate if allow_switch else None,
                allow_yield=bool(allow_switch),
                traffic_recovery=False,
            )
            # allow_yield may still be forced off inside M0 via AGV_M0_HANDOFF / meta
            if not m0_rep.get("handoff"):
                return m0_rep'''
# Actually call site should pass False when disabled:
old2 = '''                use_hierarchical=use_hierarchical,
                use_wavenet=use_wavenet,
                gate=gate if allow_switch else None,
                allow_yield=bool(allow_switch),
                traffic_recovery=False,
            )'''
new2 = '''                use_hierarchical=use_hierarchical,
                use_wavenet=use_wavenet,
                gate=(
                    gate
                    if (
                        allow_switch
                        and str(_os.environ.get("AGV_M0_HANDOFF", "1")).strip().lower()
                        not in ("0", "false", "no", "off")
                        and meta.get("m0_handoff_enabled", True) is not False
                    )
                    else None
                ),
                allow_yield=bool(
                    allow_switch
                    and str(_os.environ.get("AGV_M0_HANDOFF", "1")).strip().lower()
                    not in ("0", "false", "no", "off")
                    and meta.get("m0_handoff_enabled", True) is not False
                ),
                traffic_recovery=False,
            )'''
if text.count(old2) != 1:
    raise SystemExit(f"anchor2 count={text.count(old2)}")
text = text.replace(old2, new2, 1)
print("ok call site")

# Ensure _os is imported as alias
if "import os as _os" not in text and "import os as _os" not in text[:2000]:
    # check how os is imported
    if "import os as _os" in text:
        pass
    elif "\nimport os\n" in text[:500] or text.startswith("import os"):
        pass
    # find existing
    for line in text.splitlines()[:40]:
        if "os" in line and "import" in line:
            print("import line:", line)

ast.parse(text)
p.write_text(text, encoding="utf-8")
print("AST ok", p.stat().st_size)
