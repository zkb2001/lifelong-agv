$env:PYTHONPATH='d:\compitition\MioVerse\final_version'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:GREED_SKIP_TRAJ='1'
$env:GREED_FACN_CKPT='d:\compitition\MioVerse\final_version\greed_coop_mcts\checkpoints\facn_k_v1.pt'
Set-Location 'd:\compitition\MioVerse\final_version'
$log = 'd:\compitition\MioVerse\final_version\greed_coop_mcts\results\sh03_lifelong.log'
"==== SH03 LIFELONG FOREVER $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====" | Out-File -FilePath $log -Encoding utf8
if (-not (Test-Path 'd:\compitition\MioVerse\final_version\greed_coop_mcts\results\SH03_lifelong_seed400_task.csv')) {
  python -m greed_coop_mcts.results._gen_sh03_tasks400 *>> $log
}
# No max-total / huge wall+sim → keep streaming forever; GREED_SKIP_TRAJ=1
python -m greed_coop_mcts.solve --slot 3 --policy rrsc --device cpu --facn --lifelong-random --lifelong-seed 303 --lifelong-batch 24 --lifelong-low-water 8 --task-csv 'd:\compitition\MioVerse\final_version\greed_coop_mcts\results\SH03_lifelong_seed400_task.csv' --wall-timeout 1e12 --max-sim-time 2000000000 *>> $log
"SH03_LIFELONG_EXIT exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $log -Append -Encoding utf8
