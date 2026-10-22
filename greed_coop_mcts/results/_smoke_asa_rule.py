"""Smoke: ASA decisions on known (n, near) geometry from SH01-20 dump."""
from greed_coop_mcts.pgcp import adaptive_use_stage_net, throat_band_fleet_cap

# From production dump after ASA suite (n inferred from decisions + caps).
CASES = [
    # slot, n, near, expect_asa_net, expect_cap
    (1, 8, 14, False, 6),
    (2, 12, 45, True, 6),
    (3, 8, 74, False, 6),
    (4, 9, 54, True, 6),
    (5, 8, 62, False, 6),
    (6, 9, 37, False, 6),
    (7, 8, 67, False, 6),
    (8, 9, 108, True, 6),  # extreme labyrinth keeps StageNet
    (15, 12, 63, True, 5),  # TBC band
    (16, 15, 71, False, 6),
    (18, 18, 71, False, 6),
    (20, 30, 12, True, 7),
]


def main() -> int:
    bad = 0
    for slot, n, near, exp_asa, exp_cap in CASES:
        asa = adaptive_use_stage_net(n, near)
        cap = throat_band_fleet_cap(n, near, base=6)
        ok = asa is exp_asa and cap == exp_cap
        print(
            f"SH{slot:02d} n={n} near={near} asa={asa} (exp {exp_asa}) "
            f"cap={cap} (exp {exp_cap}) ok={ok}"
        )
        if not ok:
            bad += 1
    print("SMOKE_PASS", bad == 0)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
