# RuleNet default-off

- Default: `use_rule_net=False` → `RealtimeRuleSystem` uses **goal-flow teacher** only.
- Opt-in: `solve(..., use_rule_net=True)` or CLI `--rule-net`.
- Smoke (SH01/10/12/17): sim **identical** to prior RuleNet-on stack; wall shorter.
- Remaining learned module in default stack: **StageNet** (gated by ASA).
