def main():
    ap = argparse.ArgumentParser(description="Windowed multi-AGV MAPD. Default plan=joint (WCBS≈EECBS): all active cars planned together. max_active = parallel workers."); ap.add_argument("--slot", type=int, default=3); ap.add_argument("--max-tasks", type=int, default=0); ap.add_argument("--max-active", type=int, default=8); ap.add_argument("--plan", choices=("per_agent", "joint", "eecbs", "wcbs", "ecbs"), default="joint", help="joint/eecbs=多车联合 WCBS; per_agent=逐车优先规划"); ap.add_argument("--weight", type=float, default=1.5); ap.add_argument("--time-limit", type=float, default=35.0)
    
    ap.add_argument("--max-expansions", type=int, default=300_000)
    
    ap.add_argument("--allow-solo-fallback", action="store_true", help="允许缩到 1 车 + A*（默认关闭，强制多车联合）"); ap.add_argument("--no-initial-park", action="store_true", help="跳过开局 idle 靠边停车（可省 ~30–40 sim_t）"); ap.add_argument("--deliver-batch", type=int, default=0, help="运货子波大小；0=整波一起送（推荐配合较小 max_active）"); ap.add_argument("--turn-aware", action="store_true", help="转向感知联合规划 (x,y,pitch,t) 优先时空A*，避免事后同步转向栅栏"); ap.add_argument("--pipeline", action="store_true", help="流水线：卸完立刻派下一单 + 外圈 staging 等候，按最早到达事件截断推进")
    
    args = ap.parse_args()
    
    plan = args.plan
    
    rep = solve_ecbs(args.slot, max_tasks=args.max_tasks, max_active=args.max_active, weight=args.weight, time_limit=args.time_limit, max_expansions=args.max_expansions, plan=plan, allow_solo_fallback=bool(args.allow_solo_fallback), initial_park=not bool(args.no_initial_park), deliver_batch=int(args.deliver_batch), turn_aware=bool(args.turn_aware), pipeline=bool(args.pipeline))
    if bool(rep.get("validate_ok")):
        bool(rep.get("validate_ok"))
        if float(rep.get("completion_ratio") or 0) >= 0.999:
            float(rep.get("completion_ratio") or 0) >= 0.999
    
    ok = not rep.get("tasks_failed")
    if ok:
        return 0
    
    return 1
